// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run-only Wasmtime preparation for Forge v1.
//!
//! This module intentionally stops before loading a precompiled module. Trusted
//! deserialization is an unsafe boundary and must not be exposed until #684's
//! compilation receipt proves the exact compiled bytes and compatible runtime
//! lineage. The adapter therefore prepares a compiler-disabled engine and a
//! resource-limited/fuelled store, but provides no requester-facing load API.

#[cfg(feature = "wasm-sandbox")]
mod enabled {
    use super::super::forge_profile::{
        ForgeExecutionProfileDefinitionV1, ForgeProfileError,
    };
    use super::super::forge_wasm_features::{
        ForgeWasmFeaturePolicyError, ForgeWasmFeaturePolicyV1, ForgeWasmFeatureV1,
    };
    use std::error::Error;
    use std::fmt;
    use wasmtime::{
        Config, Engine, Store, StoreLimits, StoreLimitsBuilder, WasmFeatures,
    };

    /// A prepared Forge v1 run-only runtime.
    ///
    /// The `Engine` is deliberately private. There is no public accessor and no
    /// precompiled-module loading method in this tranche, preventing callers from
    /// treating a matching profile as permission to cross Wasmtime's unsafe
    /// deserialization boundary.
    pub struct ForgeRunOnlyRuntimeV1 {
        engine: Engine,
        profile: ForgeExecutionProfileDefinitionV1,
        profile_commitment: [u8; 32],
        feature_policy_commitment: [u8; 32],
    }

    struct ForgeStoreStateV1 {
        limits: StoreLimits,
    }

    impl ForgeRunOnlyRuntimeV1 {
        /// Prepare a compiler-disabled Wasmtime engine for one exact profile.
        ///
        /// This validates structural profile/policy invariants and requires the
        /// policy commitment to equal the commitment carried by the profile. It
        /// does not prove that the selected numeric limits are qualified, that the
        /// runtime lineage commitment matches the currently linked binary, or that
        /// any external authority permits execution.
        pub fn prepare(
            profile: ForgeExecutionProfileDefinitionV1,
            feature_policy: &ForgeWasmFeaturePolicyV1,
        ) -> Result<Self, ForgeRunOnlyRuntimeError> {
            profile.validate().map_err(ForgeRunOnlyRuntimeError::Profile)?;
            feature_policy
                .validate()
                .map_err(ForgeRunOnlyRuntimeError::FeaturePolicy)?;

            let expected_feature_policy = feature_policy
                .canonical_commitment_v1()
                .map_err(ForgeRunOnlyRuntimeError::FeaturePolicy)?;
            if profile.wasm_feature_policy_commitment != expected_feature_policy {
                return Err(ForgeRunOnlyRuntimeError::FeaturePolicyCommitmentMismatch);
            }

            let max_wasm_stack = host_usize(
                "max_wasm_stack_bytes",
                profile.max_wasm_stack_bytes,
            )?;

            let mut config = Config::new();
            // Current Symthaea's Wasmtime dependency includes a compiler through
            // default crate features. Disable it dynamically here. A future
            // qualified execution binary should additionally omit compiler crate
            // features statically and will have a different runtime-lineage
            // commitment.
            config.enable_compiler(false);
            config.consume_fuel(true);
            config.epoch_interruption(false);
            config.max_wasm_stack(max_wasm_stack);

            // Never inherit Wasmtime's evolving default feature set. Disable every
            // bit known to this exact build, then enable only stable Forge-v1 tags.
            config.wasm_features(WasmFeatures::all(), false);
            for feature in &feature_policy.allowed_features {
                config.wasm_features(wasmtime_feature(*feature), true);
            }

            let engine = Engine::new(&config)
                .map_err(|error| ForgeRunOnlyRuntimeError::Engine(error.to_string()))?;
            let profile_commitment = profile
                .canonical_commitment_v1()
                .map_err(ForgeRunOnlyRuntimeError::Profile)?;

            Ok(Self {
                engine,
                profile,
                profile_commitment,
                feature_policy_commitment: expected_feature_policy,
            })
        }

        /// Exact committed execution-profile identity prepared by this runtime.
        pub fn profile_commitment(&self) -> [u8; 32] {
            self.profile_commitment
        }

        /// Exact committed Wasm-feature policy applied to the engine.
        pub fn feature_policy_commitment(&self) -> [u8; 32] {
            self.feature_policy_commitment
        }

        /// Check only the execution-plane size ceiling for a trusted precompiled
        /// artifact candidate.
        ///
        /// Passing this check does not establish a trusted compilation receipt or
        /// make the bytes safe to deserialize.
        pub fn preflight_precompiled_artifact_len(
            &self,
            byte_len: usize,
        ) -> Result<(), ForgeRunOnlyRuntimeError> {
            let limit = host_usize(
                "max_precompiled_artifact_bytes",
                self.profile.max_precompiled_artifact_bytes,
            )?;
            if byte_len > limit {
                return Err(ForgeRunOnlyRuntimeError::PrecompiledArtifactTooLarge {
                    actual_bytes: byte_len,
                    max_bytes: limit,
                });
            }
            Ok(())
        }

        /// Create a fresh store with the exact v1 fuel and guest-resource bounds.
        ///
        /// Kept crate-private until the trusted compiled-artifact/receipt loader is
        /// implemented. Returning a Store publicly would expose its Engine and make
        /// it easier for external callers to bypass the intended deserialization
        /// choke point.
        pub(crate) fn new_store(
            &self,
        ) -> Result<Store<ForgeStoreStateV1>, ForgeRunOnlyRuntimeError> {
            let memory_size = host_usize(
                "max_linear_memory_bytes",
                self.profile.max_linear_memory_bytes,
            )?;
            let memories = host_usize("max_memories", self.profile.max_memories)?;
            let table_elements = host_usize(
                "max_table_elements",
                self.profile.max_table_elements,
            )?;
            let tables = host_usize("max_tables", self.profile.max_tables)?;
            let instances = host_usize("max_instances", self.profile.max_instances)?;

            let limits = StoreLimitsBuilder::new()
                .memory_size(memory_size)
                .memories(memories)
                .table_elements(table_elements)
                .tables(tables)
                .instances(instances)
                .trap_on_grow_failure(self.profile.trap_on_grow_failure)
                .build();

            let mut store = Store::new(&self.engine, ForgeStoreStateV1 { limits });
            store.limiter(|state| &mut state.limits);
            store
                .set_fuel(self.profile.fuel)
                .map_err(|error| ForgeRunOnlyRuntimeError::Fuel(error.to_string()))?;
            Ok(store)
        }
    }

    /// Preparation failures for the run-only Forge adapter.
    #[derive(Debug)]
    pub enum ForgeRunOnlyRuntimeError {
        Profile(ForgeProfileError),
        FeaturePolicy(ForgeWasmFeaturePolicyError),
        FeaturePolicyCommitmentMismatch,
        HostLimitWidth {
            field: &'static str,
            value: u32,
        },
        Engine(String),
        Fuel(String),
        PrecompiledArtifactTooLarge {
            actual_bytes: usize,
            max_bytes: usize,
        },
    }

    impl fmt::Display for ForgeRunOnlyRuntimeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Profile(error) => write!(f, "invalid Forge execution profile: {error}"),
                Self::FeaturePolicy(error) => write!(f, "invalid Forge feature policy: {error}"),
                Self::FeaturePolicyCommitmentMismatch => write!(
                    f,
                    "Forge feature policy commitment does not match execution profile"
                ),
                Self::HostLimitWidth { field, value } => write!(
                    f,
                    "Forge limit {field}={value} cannot be represented by host usize"
                ),
                Self::Engine(error) => write!(f, "failed to construct run-only Wasmtime engine: {error}"),
                Self::Fuel(error) => write!(f, "failed to set Forge store fuel: {error}"),
                Self::PrecompiledArtifactTooLarge {
                    actual_bytes,
                    max_bytes,
                } => write!(
                    f,
                    "precompiled Forge artifact is {actual_bytes} bytes; profile maximum is {max_bytes} bytes"
                ),
            }
        }
    }

    impl Error for ForgeRunOnlyRuntimeError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self {
                Self::Profile(error) => Some(error),
                Self::FeaturePolicy(error) => Some(error),
                _ => None,
            }
        }
    }

    fn host_usize(field: &'static str, value: u32) -> Result<usize, ForgeRunOnlyRuntimeError> {
        usize::try_from(value).map_err(|_| ForgeRunOnlyRuntimeError::HostLimitWidth {
            field,
            value,
        })
    }

    fn wasmtime_feature(feature: ForgeWasmFeatureV1) -> WasmFeatures {
        match feature {
            ForgeWasmFeatureV1::MutableGlobal => WasmFeatures::MUTABLE_GLOBAL,
            ForgeWasmFeatureV1::SaturatingFloatToInt => WasmFeatures::SATURATING_FLOAT_TO_INT,
            ForgeWasmFeatureV1::SignExtension => WasmFeatures::SIGN_EXTENSION,
            ForgeWasmFeatureV1::ReferenceTypes => WasmFeatures::REFERENCE_TYPES,
            ForgeWasmFeatureV1::MultiValue => WasmFeatures::MULTI_VALUE,
            ForgeWasmFeatureV1::BulkMemory => WasmFeatures::BULK_MEMORY,
            ForgeWasmFeatureV1::Simd => WasmFeatures::SIMD,
            ForgeWasmFeatureV1::TailCall => WasmFeatures::TAIL_CALL,
            ForgeWasmFeatureV1::Floats => WasmFeatures::FLOATS,
            ForgeWasmFeatureV1::MultiMemory => WasmFeatures::MULTI_MEMORY,
            ForgeWasmFeatureV1::ExtendedConst => WasmFeatures::EXTENDED_CONST,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::control_plane::forge::FORGE_PROTOCOL_VERSION;
        use crate::control_plane::forge_profile::{
            ForgeDeterminismPolicyV1, ForgeImportPolicyV1,
            ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
        };
        use crate::control_plane::forge_wasm_features::ForgeWasmFeatureV1;
        use wasmtime::Module;

        // Arbitrary test-vector limits. These are not operational recommendations.
        fn profile() -> ForgeExecutionProfileDefinitionV1 {
            ForgeExecutionProfileDefinitionV1 {
                protocol_version: FORGE_PROTOCOL_VERSION,
                profile_id: "adapter-test".into(),
                runtime: ForgeRuntimeIdentityV1::new_wasmtime("44.0.1", [0xA1; 32]).unwrap(),
                wasm_feature_policy_commitment: [0xB2; 32],
                abi: ForgeVerifierAbiV1::NoArgsI32,
                imports: ForgeImportPolicyV1::NoImports,
                determinism: ForgeDeterminismPolicyV1::Strict,
                interruption: ForgeInterruptionPolicyV1::DeterministicFuel,
                fuel: 10_000,
                max_precompiled_artifact_bytes: 1_000_000,
                max_linear_memory_bytes: 0,
                max_memories: 0,
                max_table_elements: 0,
                max_tables: 0,
                max_instances: 1,
                max_wasm_stack_bytes: 262_144,
                trap_on_grow_failure: true,
            }
        }

        fn bound_profile_and_policy() -> (
            ForgeExecutionProfileDefinitionV1,
            ForgeWasmFeaturePolicyV1,
        ) {
            let policy = ForgeWasmFeaturePolicyV1::new([
                ForgeWasmFeatureV1::MutableGlobal,
                ForgeWasmFeatureV1::MultiValue,
            ]);
            let mut profile = profile();
            policy.bind_execution_profile(&mut profile).unwrap();
            (profile, policy)
        }

        #[test]
        fn mismatched_feature_policy_fails_before_engine_construction() {
            let (profile, _) = bound_profile_and_policy();
            let different = ForgeWasmFeaturePolicyV1::new([]);
            assert!(matches!(
                ForgeRunOnlyRuntimeV1::prepare(profile, &different),
                Err(ForgeRunOnlyRuntimeError::FeaturePolicyCommitmentMismatch)
            ));
        }

        #[test]
        fn prepared_engine_refuses_raw_wasm_compilation() {
            let (profile, policy) = bound_profile_and_policy();
            let runtime = ForgeRunOnlyRuntimeV1::prepare(profile, &policy).unwrap();
            let minimal_wasm = b"\0asm\x01\0\0\0";
            assert!(
                Module::new(&runtime.engine, minimal_wasm).is_err(),
                "run-only Forge engine must refuse raw Wasm compilation"
            );
        }

        #[test]
        fn fresh_store_has_exact_profile_fuel() {
            let (profile, policy) = bound_profile_and_policy();
            let expected_fuel = profile.fuel;
            let runtime = ForgeRunOnlyRuntimeV1::prepare(profile, &policy).unwrap();
            let store = runtime.new_store().unwrap();
            assert_eq!(store.get_fuel().unwrap(), expected_fuel);
        }

        #[test]
        fn profile_and_feature_commitments_are_retained_exactly() {
            let (profile, policy) = bound_profile_and_policy();
            let expected_profile = profile.canonical_commitment_v1().unwrap();
            let expected_features = policy.canonical_commitment_v1().unwrap();
            let runtime = ForgeRunOnlyRuntimeV1::prepare(profile, &policy).unwrap();
            assert_eq!(runtime.profile_commitment(), expected_profile);
            assert_eq!(runtime.feature_policy_commitment(), expected_features);
        }

        #[test]
        fn precompiled_artifact_size_bound_is_admission_only() {
            let (profile, policy) = bound_profile_and_policy();
            let max = profile.max_precompiled_artifact_bytes as usize;
            let runtime = ForgeRunOnlyRuntimeV1::prepare(profile, &policy).unwrap();
            assert!(runtime.preflight_precompiled_artifact_len(max).is_ok());
            assert!(matches!(
                runtime.preflight_precompiled_artifact_len(max + 1),
                Err(ForgeRunOnlyRuntimeError::PrecompiledArtifactTooLarge { .. })
            ));
        }
    }
}

#[cfg(feature = "wasm-sandbox")]
pub use enabled::*;
