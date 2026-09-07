// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target-owned, fail-closed WebAssembly feature policy for Forge v1.
//!
//! This policy intentionally does not serialize Wasmtime/wasmparser bit values.
//! Luminous-owned stable tags describe the small v1 allowlist. A future adapter
//! must first explicitly disable every `WasmFeatures` bit known to its exact
//! Wasmtime build and then enable only the features represented here.

use super::forge::FORGE_PROTOCOL_VERSION;
use super::forge_profile::ForgeExecutionProfileDefinitionV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Domain separator for the canonical Forge v1 Wasm-feature policy.
pub const FORGE_WASM_FEATURE_POLICY_DOMAIN_V1: &[u8] =
    b"symthaea.forge.wasm-feature-policy.v1\0";

/// Feature-baseline semantics for Forge v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeWasmFeatureBaselineV1 {
    /// The adapter must start from all Wasmtime/wasmparser feature bits
    /// explicitly disabled, then enable only the committed allowlist.
    DisableAllThenAllow,
}

/// WebAssembly features that Forge v1 is willing to represent in its allowlist.
///
/// This is intentionally much narrower than Wasmtime 44.0.1's known feature
/// surface. Concurrency, relaxed-SIMD, memory64, exceptions, GC/function refs,
/// component-model features, stack switching, custom page sizes, and other
/// expanded surfaces have no v1 tag and therefore cannot be enabled by a v1
/// policy. Supporting them requires an explicit later protocol revision.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ForgeWasmFeatureV1 {
    MutableGlobal,
    SaturatingFloatToInt,
    SignExtension,
    ReferenceTypes,
    MultiValue,
    BulkMemory,
    Simd,
    TailCall,
    Floats,
    MultiMemory,
    ExtendedConst,
}

impl ForgeWasmFeatureV1 {
    /// Stable Luminous-owned canonical tag. These values are protocol surface;
    /// never replace them with Wasmtime/wasmparser internal bit values.
    pub const fn canonical_tag(self) -> u16 {
        match self {
            Self::MutableGlobal => 1,
            Self::SaturatingFloatToInt => 2,
            Self::SignExtension => 3,
            Self::ReferenceTypes => 4,
            Self::MultiValue => 5,
            Self::BulkMemory => 6,
            Self::Simd => 7,
            Self::TailCall => 8,
            Self::Floats => 9,
            Self::MultiMemory => 10,
            Self::ExtendedConst => 11,
        }
    }

    /// Exact Wasmtime 44.0.1 `WasmFeatures` constant name this v1 tag maps to.
    ///
    /// This is documentation/mapping metadata, not a dynamically resolved API.
    /// The eventual adapter must compile against and independently test the exact
    /// runtime lineage carried by the execution profile.
    pub const fn wasmtime_v44_constant_name(self) -> &'static str {
        match self {
            Self::MutableGlobal => "MUTABLE_GLOBAL",
            Self::SaturatingFloatToInt => "SATURATING_FLOAT_TO_INT",
            Self::SignExtension => "SIGN_EXTENSION",
            Self::ReferenceTypes => "REFERENCE_TYPES",
            Self::MultiValue => "MULTI_VALUE",
            Self::BulkMemory => "BULK_MEMORY",
            Self::Simd => "SIMD",
            Self::TailCall => "TAIL_CALL",
            Self::Floats => "FLOATS",
            Self::MultiMemory => "MULTI_MEMORY",
            Self::ExtendedConst => "EXTENDED_CONST",
        }
    }
}

/// Exact feature allowlist for one Forge v1 execution/compilation lineage.
///
/// An empty set is valid and means no optional `WasmFeatures` bits are enabled
/// after the adapter's disable-all step. This type does not claim that any
/// particular non-empty set is safe or qualified for production.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeWasmFeaturePolicyV1 {
    pub protocol_version: u16,
    pub baseline: ForgeWasmFeatureBaselineV1,
    pub allowed_features: BTreeSet<ForgeWasmFeatureV1>,
}

impl ForgeWasmFeaturePolicyV1 {
    /// Construct an explicit v1 set. No features are implied or inherited.
    pub fn new(
        allowed_features: impl IntoIterator<Item = ForgeWasmFeatureV1>,
    ) -> Self {
        Self {
            protocol_version: FORGE_PROTOCOL_VERSION,
            baseline: ForgeWasmFeatureBaselineV1::DisableAllThenAllow,
            allowed_features: allowed_features.into_iter().collect(),
        }
    }

    /// Validate only structural protocol invariants.
    ///
    /// This does not qualify the selected features for a workload. The selected
    /// Wasmtime runtime/config must still accept the exact set, and the Forge
    /// qualification corpus must prove the intended safety/determinism theorem.
    pub fn validate(&self) -> Result<(), ForgeWasmFeaturePolicyError> {
        if self.protocol_version != FORGE_PROTOCOL_VERSION {
            return Err(ForgeWasmFeaturePolicyError::UnsupportedProtocolVersion(
                self.protocol_version,
            ));
        }
        Ok(())
    }

    /// Stable canonical v1 bytes.
    ///
    /// Feature tags are sorted numerically even though the in-memory type is a
    /// `BTreeSet`, so canonical ordering is explicitly tied to protocol tags and
    /// not Rust enum declaration order or serde representation.
    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeWasmFeaturePolicyError> {
        self.validate()?;

        let mut tags: Vec<u16> = self
            .allowed_features
            .iter()
            .map(|feature| feature.canonical_tag())
            .collect();
        tags.sort_unstable();

        let count = u16::try_from(tags.len())
            .map_err(|_| ForgeWasmFeaturePolicyError::TooManyFeatures(tags.len()))?;
        let mut out = Vec::with_capacity(
            FORGE_WASM_FEATURE_POLICY_DOMAIN_V1.len() + 2 + 1 + 2 + tags.len() * 2,
        );
        out.extend_from_slice(FORGE_WASM_FEATURE_POLICY_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        out.push(baseline_tag(self.baseline));
        out.extend_from_slice(&count.to_be_bytes());
        for tag in tags {
            out.extend_from_slice(&tag.to_be_bytes());
        }
        Ok(out)
    }

    /// BLAKE3-256 commitment to the exact feature policy.
    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeWasmFeaturePolicyError> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }

    /// Bind this policy's exact commitment into an execution-profile definition.
    ///
    /// This mutates identity only; it does not apply the policy to a Wasmtime
    /// `Config`, prove runtime compatibility, or grant execution authority.
    pub fn bind_execution_profile(
        &self,
        profile: &mut ForgeExecutionProfileDefinitionV1,
    ) -> Result<(), ForgeWasmFeaturePolicyError> {
        profile.wasm_feature_policy_commitment = self.canonical_commitment_v1()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeWasmFeaturePolicyError {
    UnsupportedProtocolVersion(u16),
    TooManyFeatures(usize),
}

impl fmt::Display for ForgeWasmFeaturePolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProtocolVersion(version) => {
                write!(f, "unsupported Forge Wasm-feature policy version {version}")
            }
            Self::TooManyFeatures(count) => {
                write!(f, "Forge v1 Wasm-feature policy has too many features: {count}")
            }
        }
    }
}

impl Error for ForgeWasmFeaturePolicyError {}

const fn baseline_tag(value: ForgeWasmFeatureBaselineV1) -> u8 {
    match value {
        ForgeWasmFeatureBaselineV1::DisableAllThenAllow => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::forge_profile::{
        ForgeDeterminismPolicyV1, ForgeExecutionProfileDefinitionV1,
        ForgeImportPolicyV1, ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1,
        ForgeVerifierAbiV1,
    };

    fn feature(feature: ForgeWasmFeatureV1) -> BTreeSet<ForgeWasmFeatureV1> {
        BTreeSet::from([feature])
    }

    fn execution_profile() -> ForgeExecutionProfileDefinitionV1 {
        ForgeExecutionProfileDefinitionV1 {
            protocol_version: FORGE_PROTOCOL_VERSION,
            profile_id: "test-profile".into(),
            runtime: ForgeRuntimeIdentityV1::new_wasmtime("44.0.1", [0xA1; 32]).unwrap(),
            wasm_feature_policy_commitment: [0xB2; 32],
            abi: ForgeVerifierAbiV1::NoArgsI32,
            imports: ForgeImportPolicyV1::NoImports,
            determinism: ForgeDeterminismPolicyV1::Strict,
            interruption: ForgeInterruptionPolicyV1::DeterministicFuel,
            fuel: 123,
            max_precompiled_artifact_bytes: 456,
            max_linear_memory_bytes: 0,
            max_memories: 0,
            max_table_elements: 0,
            max_tables: 0,
            max_instances: 1,
            max_wasm_stack_bytes: 789,
            trap_on_grow_failure: true,
        }
    }

    #[test]
    fn empty_policy_has_frozen_disable_all_encoding() {
        let policy = ForgeWasmFeaturePolicyV1::new([]);
        let mut expected = FORGE_WASM_FEATURE_POLICY_DOMAIN_V1.to_vec();
        expected.extend_from_slice(&FORGE_PROTOCOL_VERSION.to_be_bytes());
        expected.push(1); // DisableAllThenAllow
        expected.extend_from_slice(&0u16.to_be_bytes());
        assert_eq!(policy.canonical_bytes_v1().unwrap(), expected);
    }

    #[test]
    fn canonical_tags_are_frozen_and_match_v44_names() {
        let expected = [
            (ForgeWasmFeatureV1::MutableGlobal, 1, "MUTABLE_GLOBAL"),
            (
                ForgeWasmFeatureV1::SaturatingFloatToInt,
                2,
                "SATURATING_FLOAT_TO_INT",
            ),
            (ForgeWasmFeatureV1::SignExtension, 3, "SIGN_EXTENSION"),
            (ForgeWasmFeatureV1::ReferenceTypes, 4, "REFERENCE_TYPES"),
            (ForgeWasmFeatureV1::MultiValue, 5, "MULTI_VALUE"),
            (ForgeWasmFeatureV1::BulkMemory, 6, "BULK_MEMORY"),
            (ForgeWasmFeatureV1::Simd, 7, "SIMD"),
            (ForgeWasmFeatureV1::TailCall, 8, "TAIL_CALL"),
            (ForgeWasmFeatureV1::Floats, 9, "FLOATS"),
            (ForgeWasmFeatureV1::MultiMemory, 10, "MULTI_MEMORY"),
            (ForgeWasmFeatureV1::ExtendedConst, 11, "EXTENDED_CONST"),
        ];

        for (feature, tag, name) in expected {
            assert_eq!(feature.canonical_tag(), tag);
            assert_eq!(feature.wasmtime_v44_constant_name(), name);
        }
    }

    #[test]
    fn feature_set_commitment_is_order_independent() {
        let a = ForgeWasmFeaturePolicyV1::new([
            ForgeWasmFeatureV1::BulkMemory,
            ForgeWasmFeatureV1::MultiValue,
            ForgeWasmFeatureV1::Floats,
        ]);
        let b = ForgeWasmFeaturePolicyV1::new([
            ForgeWasmFeatureV1::Floats,
            ForgeWasmFeatureV1::BulkMemory,
            ForgeWasmFeatureV1::MultiValue,
        ]);
        assert_eq!(a.canonical_commitment_v1().unwrap(), b.canonical_commitment_v1().unwrap());
    }

    #[test]
    fn every_v1_feature_is_identity_bearing() {
        let empty = ForgeWasmFeaturePolicyV1::new([])
            .canonical_commitment_v1()
            .unwrap();
        let all = [
            ForgeWasmFeatureV1::MutableGlobal,
            ForgeWasmFeatureV1::SaturatingFloatToInt,
            ForgeWasmFeatureV1::SignExtension,
            ForgeWasmFeatureV1::ReferenceTypes,
            ForgeWasmFeatureV1::MultiValue,
            ForgeWasmFeatureV1::BulkMemory,
            ForgeWasmFeatureV1::Simd,
            ForgeWasmFeatureV1::TailCall,
            ForgeWasmFeatureV1::Floats,
            ForgeWasmFeatureV1::MultiMemory,
            ForgeWasmFeatureV1::ExtendedConst,
        ];
        let mut commitments = BTreeSet::new();
        for value in all {
            let commitment = ForgeWasmFeaturePolicyV1 {
                protocol_version: FORGE_PROTOCOL_VERSION,
                baseline: ForgeWasmFeatureBaselineV1::DisableAllThenAllow,
                allowed_features: feature(value),
            }
            .canonical_commitment_v1()
            .unwrap();
            assert_ne!(commitment, empty);
            assert!(commitments.insert(commitment));
        }
    }

    #[test]
    fn binding_policy_changes_exact_execution_profile_identity() {
        let policy = ForgeWasmFeaturePolicyV1::new([
            ForgeWasmFeatureV1::MultiValue,
            ForgeWasmFeatureV1::BulkMemory,
        ]);
        let mut profile = execution_profile();
        let before = profile.canonical_commitment_v1().unwrap();
        policy.bind_execution_profile(&mut profile).unwrap();
        let after = profile.canonical_commitment_v1().unwrap();

        assert_eq!(
            profile.wasm_feature_policy_commitment,
            policy.canonical_commitment_v1().unwrap()
        );
        assert_ne!(before, after);
    }

    #[test]
    fn wrong_protocol_version_fails_closed() {
        let mut policy = ForgeWasmFeaturePolicyV1::new([]);
        policy.protocol_version += 1;
        assert!(matches!(
            policy.canonical_bytes_v1(),
            Err(ForgeWasmFeaturePolicyError::UnsupportedProtocolVersion(_))
        ));
    }
}
