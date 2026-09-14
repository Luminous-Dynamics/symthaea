// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transitive source-authority binding for runtime calibration observations.
//!
//! The already-qualified runtime observation layer binds benchmark-local source
//! inventory to effective values consumed by scoring. This module adds the next
//! provenance layer: explicit shared source authorities used by those effective-
//! value calculations.
//!
//! The authority ladder remains deliberately non-promoting:
//!
//! ```text
//! local source inventory
//!   + shared calculation authorities
//!   + runtime effective values
//!   != complete calibration census
//!   != frozen experiment
//!   != scientific validity
//! ```
//!
//! Whole-file content digests are intentionally conservative. Each dependency
//! still carries an explicit semantic role so review can distinguish why the
//! file is bound from unrelated source changes that merely force requalification.

use crate::calibration_inventory::{
    CalibrationInventory, CalibrationInventoryError, nback_source_inventory,
    stroop_source_inventory,
};
use crate::calibration_runtime::{
    RuntimeCalibrationObservation, RuntimeCalibrationObservationError, nback_runtime_observation,
    stroop_runtime_observation,
};
use crate::harness::config::BenchmarkConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CALCULATION_AUTHORITY_SCHEMA_VERSION: &str =
    "psych-calculation-authority-binding-v1";
pub const AUTHORITY_BOUND_RUNTIME_SCHEMA_VERSION: &str =
    "psych-authority-bound-runtime-observation-v1";

const CONFIG_SOURCE_PATH: &str = "src/harness/config.rs";
const DIFFICULTY_SOURCE_PATH: &str = "src/harness/difficulty.rs";
const CONFIG_SOURCE: &str = include_str!("harness/config.rs");
const DIFFICULTY_SOURCE: &str = include_str!("harness/difficulty.rs");

/// One explicitly named transitive source authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalculationDependency {
    pub dependency_id: String,
    pub source_path: String,
    pub semantic_role: String,
    pub content_digest: String,
}

impl CalculationDependency {
    pub fn from_source(
        dependency_id: impl Into<String>,
        source_path: impl Into<String>,
        semantic_role: impl Into<String>,
        source: &str,
    ) -> Self {
        Self {
            dependency_id: dependency_id.into(),
            source_path: source_path.into(),
            semantic_role: semantic_role.into(),
            content_digest: blake3::hash(source.as_bytes()).to_hex().to_string(),
        }
    }

    fn validate(&self) -> Result<(), CalculationAuthorityError> {
        if self.dependency_id.trim().is_empty()
            || self.source_path.trim().is_empty()
            || self.semantic_role.trim().is_empty()
        {
            return Err(CalculationAuthorityError::EmptyDependencyField);
        }
        if self.content_digest.len() != 64
            || !self
                .content_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(CalculationAuthorityError::InvalidContentDigest);
        }
        Ok(())
    }
}

/// Explicit transitive authority set for one benchmark's migrated calculation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalculationAuthorityBinding {
    pub schema_version: String,
    pub benchmark: String,
    pub revision: u32,
    pub local_source_inventory_digest: String,
    pub dependencies: Vec<CalculationDependency>,
}

impl CalculationAuthorityBinding {
    pub fn new(
        benchmark: impl Into<String>,
        revision: u32,
        local_source_inventory_digest: impl Into<String>,
        dependencies: Vec<CalculationDependency>,
    ) -> Self {
        Self {
            schema_version: CALCULATION_AUTHORITY_SCHEMA_VERSION.to_string(),
            benchmark: benchmark.into(),
            revision,
            local_source_inventory_digest: local_source_inventory_digest.into(),
            dependencies,
        }
    }

    pub fn validate(&self) -> Result<(), CalculationAuthorityError> {
        if self.schema_version != CALCULATION_AUTHORITY_SCHEMA_VERSION {
            return Err(CalculationAuthorityError::UnsupportedSchema);
        }
        if self.benchmark.trim().is_empty() {
            return Err(CalculationAuthorityError::EmptyBenchmark);
        }
        if self.revision == 0 {
            return Err(CalculationAuthorityError::InvalidRevision);
        }
        if self.local_source_inventory_digest.trim().is_empty() {
            return Err(CalculationAuthorityError::EmptyLocalInventoryDigest);
        }
        if self.dependencies.is_empty() {
            return Err(CalculationAuthorityError::EmptyDependencies);
        }

        let mut ids = BTreeSet::new();
        for dependency in &self.dependencies {
            dependency.validate()?;
            if !ids.insert(dependency.dependency_id.as_str()) {
                return Err(CalculationAuthorityError::DuplicateDependencyId);
            }
        }
        Ok(())
    }

    pub fn validate_against_inventory(
        &self,
        inventory: &CalibrationInventory,
    ) -> Result<(), CalculationAuthorityError> {
        self.validate()?;
        inventory.validate()?;
        if self.benchmark != inventory.benchmark {
            return Err(CalculationAuthorityError::BenchmarkMismatch);
        }
        if self.local_source_inventory_digest != inventory.digest_hex()? {
            return Err(CalculationAuthorityError::LocalInventoryDigestMismatch);
        }
        Ok(())
    }

    /// Stable identity independent of dependency declaration order.
    pub fn digest_hex(&self) -> Result<String, CalculationAuthorityError> {
        self.validate()?;
        let mut dependencies = self.dependencies.iter().collect::<Vec<_>>();
        dependencies.sort_by(|left, right| left.dependency_id.cmp(&right.dependency_id));

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.calculation-authority-binding.v1\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.benchmark.as_bytes());
        hasher.update(&self.revision.to_le_bytes());
        hash_field(&mut hasher, self.local_source_inventory_digest.as_bytes());
        hasher.update(&(dependencies.len() as u64).to_le_bytes());
        for dependency in dependencies {
            hash_field(&mut hasher, dependency.dependency_id.as_bytes());
            hash_field(&mut hasher, dependency.source_path.as_bytes());
            hash_field(&mut hasher, dependency.semantic_role.as_bytes());
            hash_field(&mut hasher, dependency.content_digest.as_bytes());
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

/// Stronger wrapper around the previously qualified runtime observation.
///
/// The nested observation retains its original schema and meaning. This wrapper
/// adds transitive source-lineage identity without rewriting prior evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityBoundRuntimeCalibrationObservation {
    pub schema_version: String,
    pub runtime_observation: RuntimeCalibrationObservation,
    pub calculation_authority_digest: String,
}

impl AuthorityBoundRuntimeCalibrationObservation {
    pub fn new(
        runtime_observation: RuntimeCalibrationObservation,
        inventory: &CalibrationInventory,
        authority: &CalculationAuthorityBinding,
    ) -> Result<Self, AuthorityBoundRuntimeError> {
        runtime_observation.validate_against_inventory(inventory)?;
        authority.validate_against_inventory(inventory)?;
        if runtime_observation.benchmark != authority.benchmark {
            return Err(AuthorityBoundRuntimeError::BenchmarkMismatch);
        }

        Ok(Self {
            schema_version: AUTHORITY_BOUND_RUNTIME_SCHEMA_VERSION.to_string(),
            runtime_observation,
            calculation_authority_digest: authority.digest_hex()?,
        })
    }

    pub fn validate(
        &self,
        inventory: &CalibrationInventory,
        authority: &CalculationAuthorityBinding,
    ) -> Result<(), AuthorityBoundRuntimeError> {
        if self.schema_version != AUTHORITY_BOUND_RUNTIME_SCHEMA_VERSION {
            return Err(AuthorityBoundRuntimeError::UnsupportedSchema);
        }
        self.runtime_observation
            .validate_against_inventory(inventory)?;
        authority.validate_against_inventory(inventory)?;
        if self.runtime_observation.benchmark != authority.benchmark {
            return Err(AuthorityBoundRuntimeError::BenchmarkMismatch);
        }
        if self.calculation_authority_digest != authority.digest_hex()? {
            return Err(AuthorityBoundRuntimeError::AuthorityDigestMismatch);
        }
        Ok(())
    }

    pub fn digest_hex(&self) -> Result<String, AuthorityBoundRuntimeError> {
        if self.schema_version != AUTHORITY_BOUND_RUNTIME_SCHEMA_VERSION
            || self.calculation_authority_digest.trim().is_empty()
        {
            return Err(AuthorityBoundRuntimeError::IncompleteObservation);
        }
        let runtime_digest = self.runtime_observation.digest_hex()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.authority-bound-runtime-observation.v1\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, runtime_digest.as_bytes());
        hash_field(&mut hasher, self.calculation_authority_digest.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

pub fn stroop_calculation_authority_binding(
) -> Result<CalculationAuthorityBinding, CalculationAuthorityError> {
    let inventory = stroop_source_inventory()?;
    let binding = CalculationAuthorityBinding::new(
        inventory.benchmark.clone(),
        1,
        inventory.digest_hex()?,
        vec![
            CalculationDependency::from_source(
                "benchmark_config.effective_noise",
                CONFIG_SOURCE_PATH,
                "BenchmarkConfig::effective_noise combines encoding_noise and time_pressure for the migrated scoring path",
                CONFIG_SOURCE,
            ),
            CalculationDependency::from_source(
                "difficulty.stroop_selection_and_transforms",
                DIFFICULTY_SOURCE_PATH,
                "difficulty_model_for(Executive::Stroop) plus interference_multiplier and temperature_multiplier",
                DIFFICULTY_SOURCE,
            ),
        ],
    );
    binding.validate_against_inventory(&inventory)?;
    Ok(binding)
}

pub fn nback_calculation_authority_binding(
) -> Result<CalculationAuthorityBinding, CalculationAuthorityError> {
    let inventory = nback_source_inventory()?;
    let binding = CalculationAuthorityBinding::new(
        inventory.benchmark.clone(),
        1,
        inventory.digest_hex()?,
        vec![
            CalculationDependency::from_source(
                "benchmark_config.effective_noise",
                CONFIG_SOURCE_PATH,
                "BenchmarkConfig::effective_noise combines encoding_noise and time_pressure for the migrated threshold path",
                CONFIG_SOURCE,
            ),
            CalculationDependency::from_source(
                "difficulty.nback_selection_and_temperature",
                DIFFICULTY_SOURCE_PATH,
                "difficulty_model_for(WorM::N-back) plus temperature_multiplier used by the migrated threshold helper",
                DIFFICULTY_SOURCE,
            ),
        ],
    );
    binding.validate_against_inventory(&inventory)?;
    Ok(binding)
}

pub fn stroop_authority_bound_runtime_observation(
    config: &BenchmarkConfig,
) -> Result<AuthorityBoundRuntimeCalibrationObservation, AuthorityBoundRuntimeError> {
    let inventory = stroop_source_inventory()?;
    let authority = stroop_calculation_authority_binding()?;
    let runtime = stroop_runtime_observation(config)?;
    AuthorityBoundRuntimeCalibrationObservation::new(runtime, &inventory, &authority)
}

pub fn nback_authority_bound_runtime_observation(
    config: &BenchmarkConfig,
) -> Result<AuthorityBoundRuntimeCalibrationObservation, AuthorityBoundRuntimeError> {
    let inventory = nback_source_inventory()?;
    let authority = nback_calculation_authority_binding()?;
    let runtime = nback_runtime_observation(config)?;
    AuthorityBoundRuntimeCalibrationObservation::new(runtime, &inventory, &authority)
}

pub fn validate_stroop_authority_bound_against_config(
    observation: &AuthorityBoundRuntimeCalibrationObservation,
    config: &BenchmarkConfig,
) -> Result<(), AuthorityBoundRuntimeError> {
    let inventory = stroop_source_inventory()?;
    let authority = stroop_calculation_authority_binding()?;
    observation.validate(&inventory, &authority)?;
    let expected = stroop_authority_bound_runtime_observation(config)?;
    if observation != &expected {
        return Err(AuthorityBoundRuntimeError::RuntimeObservationMismatch);
    }
    Ok(())
}

pub fn validate_nback_authority_bound_against_config(
    observation: &AuthorityBoundRuntimeCalibrationObservation,
    config: &BenchmarkConfig,
) -> Result<(), AuthorityBoundRuntimeError> {
    let inventory = nback_source_inventory()?;
    let authority = nback_calculation_authority_binding()?;
    observation.validate(&inventory, &authority)?;
    let expected = nback_authority_bound_runtime_observation(config)?;
    if observation != &expected {
        return Err(AuthorityBoundRuntimeError::RuntimeObservationMismatch);
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalculationAuthorityError {
    Inventory(CalibrationInventoryError),
    UnsupportedSchema,
    EmptyBenchmark,
    InvalidRevision,
    EmptyLocalInventoryDigest,
    EmptyDependencies,
    EmptyDependencyField,
    DuplicateDependencyId,
    InvalidContentDigest,
    BenchmarkMismatch,
    LocalInventoryDigestMismatch,
}

impl From<CalibrationInventoryError> for CalculationAuthorityError {
    fn from(value: CalibrationInventoryError) -> Self {
        Self::Inventory(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityBoundRuntimeError {
    Inventory(CalibrationInventoryError),
    Authority(CalculationAuthorityError),
    Runtime(RuntimeCalibrationObservationError),
    UnsupportedSchema,
    BenchmarkMismatch,
    AuthorityDigestMismatch,
    IncompleteObservation,
    RuntimeObservationMismatch,
}

impl From<CalibrationInventoryError> for AuthorityBoundRuntimeError {
    fn from(value: CalibrationInventoryError) -> Self {
        Self::Inventory(value)
    }
}

impl From<CalculationAuthorityError> for AuthorityBoundRuntimeError {
    fn from(value: CalculationAuthorityError) -> Self {
        Self::Authority(value)
    }
}

impl From<RuntimeCalibrationObservationError> for AuthorityBoundRuntimeError {
    fn from(value: RuntimeCalibrationObservationError) -> Self {
        Self::Runtime(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_binding(source: &str) -> CalculationAuthorityBinding {
        CalculationAuthorityBinding::new(
            "Synthetic",
            1,
            "local-inventory-digest",
            vec![CalculationDependency::from_source(
                "shared.rule",
                "src/shared.rs",
                "synthetic shared calculation rule",
                source,
            )],
        )
    }

    #[test]
    fn stroop_binding_names_explicit_transitive_authorities() {
        let binding = stroop_calculation_authority_binding().unwrap();
        let inventory = stroop_source_inventory().unwrap();
        assert!(binding.validate_against_inventory(&inventory).is_ok());
        assert_eq!(binding.dependencies.len(), 2);
        assert_eq!(
            binding.dependencies[0].dependency_id,
            "benchmark_config.effective_noise"
        );
        assert_eq!(binding.dependencies[0].source_path, CONFIG_SOURCE_PATH);
        assert_eq!(binding.dependencies[1].source_path, DIFFICULTY_SOURCE_PATH);
    }

    #[test]
    fn nback_binding_names_explicit_transitive_authorities() {
        let binding = nback_calculation_authority_binding().unwrap();
        let inventory = nback_source_inventory().unwrap();
        assert!(binding.validate_against_inventory(&inventory).is_ok());
        assert_eq!(binding.dependencies.len(), 2);
        assert_eq!(
            binding.dependencies[1].dependency_id,
            "difficulty.nback_selection_and_temperature"
        );
    }

    #[test]
    fn dependency_order_does_not_change_binding_digest() {
        let first = stroop_calculation_authority_binding().unwrap();
        let mut second = first.clone();
        second.dependencies.reverse();
        assert_eq!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn shared_source_change_changes_authority_digest() {
        let first = synthetic_binding("fn shared() -> f64 { 0.5 }");
        let second = synthetic_binding("fn shared() -> f64 { 0.6 }");
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn semantic_role_change_changes_authority_digest() {
        let first = synthetic_binding("same source");
        let mut second = first.clone();
        second.dependencies[0].semantic_role = "different reviewed role".to_string();
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn local_inventory_change_changes_authority_digest() {
        let first = synthetic_binding("same source");
        let mut second = first.clone();
        second.local_source_inventory_digest = "different-local-inventory".to_string();
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
    }

    #[test]
    fn duplicate_dependency_ids_fail_closed() {
        let mut binding = synthetic_binding("same source");
        binding.dependencies.push(binding.dependencies[0].clone());
        assert_eq!(
            binding.validate(),
            Err(CalculationAuthorityError::DuplicateDependencyId)
        );
    }

    #[test]
    fn empty_dependency_identity_fails_closed() {
        let mut binding = synthetic_binding("same source");
        binding.dependencies[0].dependency_id.clear();
        assert_eq!(
            binding.validate(),
            Err(CalculationAuthorityError::EmptyDependencyField)
        );
    }

    #[test]
    fn stroop_bound_observation_revalidates_current_sources_and_config() {
        let config = BenchmarkConfig::default();
        let observation = stroop_authority_bound_runtime_observation(&config).unwrap();
        assert!(validate_stroop_authority_bound_against_config(&observation, &config).is_ok());
        assert_eq!(
            observation.calculation_authority_digest,
            stroop_calculation_authority_binding()
                .unwrap()
                .digest_hex()
                .unwrap()
        );
    }

    #[test]
    fn nback_bound_observation_revalidates_current_sources_and_config() {
        let mut config = BenchmarkConfig::default();
        config.encoding_noise = 0.2;
        config.time_pressure = 0.3;
        config.difficulty = 0.6;
        let observation = nback_authority_bound_runtime_observation(&config).unwrap();
        assert!(validate_nback_authority_bound_against_config(&observation, &config).is_ok());
    }

    #[test]
    fn tampered_authority_digest_fails_before_runtime_comparison() {
        let config = BenchmarkConfig::default();
        let mut observation = stroop_authority_bound_runtime_observation(&config).unwrap();
        observation.calculation_authority_digest = "0".repeat(64);
        assert_eq!(
            validate_stroop_authority_bound_against_config(&observation, &config),
            Err(AuthorityBoundRuntimeError::AuthorityDigestMismatch)
        );
    }

    #[test]
    fn serialization_preserves_transitive_binding() {
        let config = BenchmarkConfig::default();
        let observation = nback_authority_bound_runtime_observation(&config).unwrap();
        let json = serde_json::to_string(&observation).unwrap();
        let decoded: AuthorityBoundRuntimeCalibrationObservation =
            serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, observation);
        assert!(validate_nback_authority_bound_against_config(&decoded, &config).is_ok());
    }
}
