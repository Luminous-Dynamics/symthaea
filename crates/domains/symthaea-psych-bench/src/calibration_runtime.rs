// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-authoritative runtime observations for calibration-relevant values.
//!
//! A source inventory proves which rule/expression exists. This layer records
//! effective values computed by the *same helper consumed by benchmark scoring*.
//! It deliberately exposes no promotion into `CalibrationManifest` or frozen
//! authority.

use crate::benchmarks::{executive::StroopBenchmark, worm::NBackBenchmark};
use crate::calibration_inventory::{
    CalibrationInventory, CalibrationInventoryCoverage, CalibrationInventoryError,
    nback_source_inventory, stroop_source_inventory,
};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RUNTIME_CALIBRATION_OBSERVATION_SCHEMA_VERSION: &str =
    "psych-runtime-calibration-observation-v1";

/// Runtime-surface coverage is explicitly non-authoritative in this tranche.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCalibrationCoverage {
    Partial,
}

/// Exact runtime observation for a migrated calibration surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCalibrationObservation {
    pub schema_version: String,
    pub benchmark: String,
    pub source_inventory_digest: String,
    pub source_inventory_coverage: CalibrationInventoryCoverage,
    pub runtime_coverage: RuntimeCalibrationCoverage,
    /// Relevant inputs to the effective-value calculation, canonically encoded.
    pub config_inputs: BTreeMap<String, String>,
    /// Effective values actually consumed by the migrated scoring path.
    pub effective_values: BTreeMap<String, String>,
}

impl RuntimeCalibrationObservation {
    /// Validate only the source/provenance side of the observation.
    ///
    /// This does not establish that the stored effective values match a caller's
    /// runtime configuration. Use the benchmark-specific config validator for that.
    pub fn validate_against_inventory(
        &self,
        inventory: &CalibrationInventory,
    ) -> Result<(), RuntimeCalibrationObservationError> {
        if self.schema_version != RUNTIME_CALIBRATION_OBSERVATION_SCHEMA_VERSION {
            return Err(RuntimeCalibrationObservationError::UnsupportedSchema);
        }
        inventory.validate()?;
        if self.benchmark != inventory.benchmark {
            return Err(RuntimeCalibrationObservationError::BenchmarkMismatch);
        }
        if self.source_inventory_coverage != inventory.coverage {
            return Err(RuntimeCalibrationObservationError::InventoryCoverageMismatch);
        }
        if self.source_inventory_digest != inventory.digest_hex()? {
            return Err(RuntimeCalibrationObservationError::InventoryDigestMismatch);
        }
        if self.config_inputs.is_empty() || self.effective_values.is_empty() {
            return Err(RuntimeCalibrationObservationError::IncompleteObservation);
        }
        Ok(())
    }

    /// Canonical observation digest. `BTreeMap` ordering makes the mapping
    /// stable independent of insertion order.
    pub fn digest_hex(&self) -> Result<String, RuntimeCalibrationObservationError> {
        if self.schema_version != RUNTIME_CALIBRATION_OBSERVATION_SCHEMA_VERSION {
            return Err(RuntimeCalibrationObservationError::UnsupportedSchema);
        }
        if self.benchmark.trim().is_empty()
            || self.source_inventory_digest.trim().is_empty()
            || self.config_inputs.is_empty()
            || self.effective_values.is_empty()
        {
            return Err(RuntimeCalibrationObservationError::IncompleteObservation);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.runtime-calibration-observation.v1\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.benchmark.as_bytes());
        hash_field(&mut hasher, self.source_inventory_digest.as_bytes());
        hasher.update(&[match self.source_inventory_coverage {
            CalibrationInventoryCoverage::Partial => 1,
            CalibrationInventoryCoverage::SourceEnumerated => 2,
        }]);
        hasher.update(&[match self.runtime_coverage {
            RuntimeCalibrationCoverage::Partial => 1,
        }]);
        hash_map(&mut hasher, &self.config_inputs);
        hash_map(&mut hasher, &self.effective_values);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_common_inputs(
    config: &BenchmarkConfig,
) -> Result<(), RuntimeCalibrationObservationError> {
    if !config.difficulty.is_finite()
        || !config.encoding_noise.is_finite()
        || !config.time_pressure.is_finite()
    {
        return Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue);
    }
    Ok(())
}

fn common_config_inputs(config: &BenchmarkConfig) -> BTreeMap<String, String> {
    let mut config_inputs = BTreeMap::new();
    config_inputs.insert("difficulty".to_string(), canonical_f64(config.difficulty));
    config_inputs.insert(
        "encoding_noise".to_string(),
        canonical_f64(config.encoding_noise),
    );
    config_inputs.insert(
        "time_pressure".to_string(),
        canonical_f64(config.time_pressure),
    );
    config_inputs
}

/// Observe the exact migrated Stroop effective values used by scoring for this
/// configuration. The benchmark and this observer call the same calculation
/// helper; there is no duplicated calibration formula here.
pub fn stroop_runtime_observation(
    config: &BenchmarkConfig,
) -> Result<RuntimeCalibrationObservation, RuntimeCalibrationObservationError> {
    validate_common_inputs(config)?;

    let inventory = stroop_source_inventory()?;
    let diff_model = difficulty_model_for("Executive::Stroop");
    let values = StroopBenchmark::effective_calibration_values(config, &diff_model);
    if !values.reading_automaticity.is_finite()
        || !values.temperature.is_finite()
        || !values.encoding_noise.is_finite()
    {
        return Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue);
    }

    let mut effective_values = BTreeMap::new();
    effective_values.insert(
        "effective_encoding_noise".to_string(),
        canonical_f32(values.encoding_noise),
    );
    effective_values.insert(
        "reading_automaticity".to_string(),
        canonical_f32(values.reading_automaticity),
    );
    effective_values.insert(
        "temperature".to_string(),
        canonical_f64(values.temperature),
    );

    let observation = RuntimeCalibrationObservation {
        schema_version: RUNTIME_CALIBRATION_OBSERVATION_SCHEMA_VERSION.to_string(),
        benchmark: inventory.benchmark.clone(),
        source_inventory_digest: inventory.digest_hex()?,
        source_inventory_coverage: inventory.coverage,
        runtime_coverage: RuntimeCalibrationCoverage::Partial,
        config_inputs: common_config_inputs(config),
        effective_values,
    };
    observation.validate_against_inventory(&inventory)?;
    Ok(observation)
}

/// Recompute the Stroop observation through the same helper used by scoring and
/// require exact equality with a stored/deserialized receipt.
pub fn validate_stroop_against_config(
    observation: &RuntimeCalibrationObservation,
    config: &BenchmarkConfig,
) -> Result<(), RuntimeCalibrationObservationError> {
    let expected = stroop_runtime_observation(config)?;
    if observation != &expected {
        return Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch);
    }
    Ok(())
}

/// Observe the exact migrated N-back threshold values consumed by scoring.
///
/// Runtime coverage remains partial: this binds the executed threshold chain,
/// not a claim that all upstream scientific/calibration choices are complete.
pub fn nback_runtime_observation(
    config: &BenchmarkConfig,
) -> Result<RuntimeCalibrationObservation, RuntimeCalibrationObservationError> {
    validate_common_inputs(config)?;

    let inventory = nback_source_inventory()?;
    let diff_model = difficulty_model_for("WorM::N-back");
    let temp_mult = diff_model.temperature_multiplier(config.difficulty);
    let values = NBackBenchmark::effective_calibration_values(config, temp_mult);

    if !values.effective_noise.is_finite()
        || !values.temperature_multiplier.is_finite()
        || !values.base_threshold.is_finite()
        || !values.match_threshold.is_finite()
    {
        return Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue);
    }

    let mut effective_values = BTreeMap::new();
    effective_values.insert(
        "effective_encoding_noise".to_string(),
        canonical_f64(values.effective_noise),
    );
    effective_values.insert(
        "temperature_multiplier".to_string(),
        canonical_f64(values.temperature_multiplier),
    );
    effective_values.insert(
        "base_threshold".to_string(),
        canonical_f64(values.base_threshold),
    );
    effective_values.insert(
        "match_threshold".to_string(),
        canonical_f32(values.match_threshold),
    );

    let observation = RuntimeCalibrationObservation {
        schema_version: RUNTIME_CALIBRATION_OBSERVATION_SCHEMA_VERSION.to_string(),
        benchmark: inventory.benchmark.clone(),
        source_inventory_digest: inventory.digest_hex()?,
        source_inventory_coverage: inventory.coverage,
        runtime_coverage: RuntimeCalibrationCoverage::Partial,
        config_inputs: common_config_inputs(config),
        effective_values,
    };
    observation.validate_against_inventory(&inventory)?;
    Ok(observation)
}

/// Recompute the N-back observation through the same helper used by scoring and
/// require exact equality with a stored/deserialized receipt.
pub fn validate_nback_against_config(
    observation: &RuntimeCalibrationObservation,
    config: &BenchmarkConfig,
) -> Result<(), RuntimeCalibrationObservationError> {
    let expected = nback_runtime_observation(config)?;
    if observation != &expected {
        return Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch);
    }
    Ok(())
}

fn canonical_f64(value: f64) -> String {
    format!("{value:.17e}|bits=0x{:016x}", value.to_bits())
}

fn canonical_f32(value: f32) -> String {
    format!("{value:.9e}|bits=0x{:08x}", value.to_bits())
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_map(hasher: &mut blake3::Hasher, map: &BTreeMap<String, String>) {
    hasher.update(&(map.len() as u64).to_le_bytes());
    for (key, value) in map {
        hash_field(hasher, key.as_bytes());
        hash_field(hasher, value.as_bytes());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCalibrationObservationError {
    Inventory(CalibrationInventoryError),
    UnsupportedSchema,
    BenchmarkMismatch,
    InventoryCoverageMismatch,
    InventoryDigestMismatch,
    IncompleteObservation,
    RuntimeObservationMismatch,
    NonFiniteInputOrValue,
}

impl From<CalibrationInventoryError> for RuntimeCalibrationObservationError {
    fn from(value: CalibrationInventoryError) -> Self {
        Self::Inventory(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stroop_observation_binds_exact_effective_values() {
        let config = BenchmarkConfig::default();
        let observation = stroop_runtime_observation(&config).unwrap();
        assert_eq!(observation.benchmark, "Executive::Stroop");
        assert_eq!(observation.runtime_coverage, RuntimeCalibrationCoverage::Partial);
        assert_eq!(
            observation.effective_values["reading_automaticity"],
            canonical_f32(0.35)
        );
        assert_eq!(
            observation.effective_values["temperature"],
            canonical_f64(0.25)
        );
        assert_eq!(
            observation.effective_values["effective_encoding_noise"],
            canonical_f32(0.0)
        );
        assert!(
            observation
                .validate_against_inventory(&stroop_source_inventory().unwrap())
                .is_ok()
        );
        assert!(validate_stroop_against_config(&observation, &config).is_ok());
    }

    #[test]
    fn shared_helper_is_bit_identical_to_pre_refactor_stroop_formula() {
        let mut cases = vec![BenchmarkConfig::default()];

        let mut pressured = BenchmarkConfig::default();
        pressured.time_pressure = 0.4;
        cases.push(pressured);

        let mut difficult = BenchmarkConfig::default();
        difficult.difficulty = 0.7;
        cases.push(difficult);

        let mut noisy = BenchmarkConfig::default();
        noisy.encoding_noise = 0.23;
        noisy.time_pressure = 0.35;
        noisy.difficulty = 0.55;
        cases.push(noisy);

        for config in cases {
            let diff_model = difficulty_model_for("Executive::Stroop");
            let actual = StroopBenchmark::effective_calibration_values(&config, &diff_model);

            // Migration oracle: exact pre-refactor production expressions.
            // This duplication is test-only and must never become a second
            // production calculation path.
            let base_automaticity: f32 = 0.35;
            let expected_reading = (base_automaticity
                * diff_model.interference_multiplier(config.difficulty) as f32)
                .min(0.95);
            let base_temperature: f64 = 0.25 + config.time_pressure * 0.15;
            let expected_temperature =
                base_temperature * diff_model.temperature_multiplier(config.difficulty);
            let expected_noise = config.effective_noise() as f32;

            assert_eq!(actual.reading_automaticity.to_bits(), expected_reading.to_bits());
            assert_eq!(actual.temperature.to_bits(), expected_temperature.to_bits());
            assert_eq!(actual.encoding_noise.to_bits(), expected_noise.to_bits());
        }
    }

    #[test]
    fn default_nback_observation_binds_exact_threshold_chain() {
        let config = BenchmarkConfig::default();
        let observation = nback_runtime_observation(&config).unwrap();
        let diff_model = difficulty_model_for("WorM::N-back");
        let temp_mult = diff_model.temperature_multiplier(config.difficulty);
        let values = NBackBenchmark::effective_calibration_values(&config, temp_mult);

        assert_eq!(observation.benchmark, "WorM::N-back");
        assert_eq!(observation.runtime_coverage, RuntimeCalibrationCoverage::Partial);
        assert_eq!(
            observation.effective_values["effective_encoding_noise"],
            canonical_f64(values.effective_noise)
        );
        assert_eq!(
            observation.effective_values["temperature_multiplier"],
            canonical_f64(values.temperature_multiplier)
        );
        assert_eq!(
            observation.effective_values["base_threshold"],
            canonical_f64(values.base_threshold)
        );
        assert_eq!(
            observation.effective_values["match_threshold"],
            canonical_f32(values.match_threshold)
        );
        assert!(
            observation
                .validate_against_inventory(&nback_source_inventory().unwrap())
                .is_ok()
        );
        assert!(validate_nback_against_config(&observation, &config).is_ok());
    }

    #[test]
    fn shared_helper_is_bit_identical_to_pre_refactor_nback_formula() {
        let mut cases = vec![BenchmarkConfig::default()];

        let mut pressured = BenchmarkConfig::default();
        pressured.time_pressure = 0.4;
        cases.push(pressured);

        let mut difficult = BenchmarkConfig::default();
        difficult.difficulty = 0.7;
        cases.push(difficult);

        let mut noisy = BenchmarkConfig::default();
        noisy.encoding_noise = 0.23;
        noisy.time_pressure = 0.35;
        noisy.difficulty = 0.55;
        cases.push(noisy);

        for config in cases {
            let diff_model = difficulty_model_for("WorM::N-back");
            let temp_mult = diff_model.temperature_multiplier(config.difficulty);
            let actual = NBackBenchmark::effective_calibration_values(&config, temp_mult);

            // Migration oracle: exact pre-refactor production expressions.
            // Duplication is test-only; production has one calculation authority.
            let expected_noise = config.effective_noise();
            let expected_base = 0.50 - config.time_pressure * 0.25 - expected_noise * 0.15;
            let expected_match = (expected_base / temp_mult).max(0.05) as f32;

            assert_eq!(actual.effective_noise.to_bits(), expected_noise.to_bits());
            assert_eq!(actual.temperature_multiplier.to_bits(), temp_mult.to_bits());
            assert_eq!(actual.base_threshold.to_bits(), expected_base.to_bits());
            assert_eq!(actual.match_threshold.to_bits(), expected_match.to_bits());
        }
    }

    #[test]
    fn relevant_config_change_changes_runtime_observation_digest() {
        let base = BenchmarkConfig::default();
        let mut changed = base.clone();
        changed.time_pressure = 0.5;
        let first = stroop_runtime_observation(&base).unwrap();
        let second = stroop_runtime_observation(&changed).unwrap();
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
        assert_ne!(
            first.effective_values["temperature"],
            second.effective_values["temperature"]
        );
        assert_eq!(
            validate_stroop_against_config(&first, &changed),
            Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch)
        );
    }

    #[test]
    fn nback_relevant_config_change_changes_runtime_observation_digest() {
        let base = BenchmarkConfig::default();
        let mut changed = base.clone();
        changed.encoding_noise = 0.3;
        let first = nback_runtime_observation(&base).unwrap();
        let second = nback_runtime_observation(&changed).unwrap();

        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
        assert_ne!(
            first.effective_values["effective_encoding_noise"],
            second.effective_values["effective_encoding_noise"]
        );
        assert_ne!(
            first.effective_values["match_threshold"],
            second.effective_values["match_threshold"]
        );
        assert_eq!(
            validate_nback_against_config(&first, &changed),
            Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch)
        );
    }

    #[test]
    fn non_finite_inputs_do_not_produce_evidence_receipts() {
        let mut nan_difficulty = BenchmarkConfig::default();
        nan_difficulty.difficulty = f64::NAN;
        assert_eq!(
            stroop_runtime_observation(&nan_difficulty),
            Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue)
        );
        assert_eq!(
            nback_runtime_observation(&nan_difficulty),
            Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue)
        );

        let mut infinite_noise = BenchmarkConfig::default();
        infinite_noise.encoding_noise = f64::INFINITY;
        assert_eq!(
            stroop_runtime_observation(&infinite_noise),
            Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue)
        );
        assert_eq!(
            nback_runtime_observation(&infinite_noise),
            Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue)
        );

        let mut infinite_pressure = BenchmarkConfig::default();
        infinite_pressure.time_pressure = f64::NEG_INFINITY;
        assert_eq!(
            stroop_runtime_observation(&infinite_pressure),
            Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue)
        );
        assert_eq!(
            nback_runtime_observation(&infinite_pressure),
            Err(RuntimeCalibrationObservationError::NonFiniteInputOrValue)
        );
    }

    #[test]
    fn tampered_inventory_digest_fails_revalidation() {
        let config = BenchmarkConfig::default();
        let inventory = stroop_source_inventory().unwrap();
        let mut observation = stroop_runtime_observation(&config).unwrap();
        observation.source_inventory_digest = "tampered".to_string();
        assert_eq!(
            observation.validate_against_inventory(&inventory),
            Err(RuntimeCalibrationObservationError::InventoryDigestMismatch)
        );
        assert_eq!(
            validate_stroop_against_config(&observation, &config),
            Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch)
        );
    }

    #[test]
    fn changing_one_effective_value_fails_runtime_revalidation() {
        let config = BenchmarkConfig::default();
        let first = stroop_runtime_observation(&config).unwrap();
        let mut second = first.clone();
        second.effective_values.insert(
            "temperature".to_string(),
            canonical_f64(0.25000000000000006),
        );
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
        assert_eq!(
            validate_stroop_against_config(&second, &config),
            Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch)
        );
    }

    #[test]
    fn changing_nback_effective_value_fails_runtime_revalidation() {
        let config = BenchmarkConfig::default();
        let first = nback_runtime_observation(&config).unwrap();
        let mut second = first.clone();
        second.effective_values.insert(
            "match_threshold".to_string(),
            canonical_f32(0.50000006),
        );
        assert_ne!(first.digest_hex().unwrap(), second.digest_hex().unwrap());
        assert_eq!(
            validate_nback_against_config(&second, &config),
            Err(RuntimeCalibrationObservationError::RuntimeObservationMismatch)
        );
    }

    #[test]
    fn exact_float_bits_survive_serialization_round_trip() {
        let config = BenchmarkConfig::default();
        for observation in [
            stroop_runtime_observation(&config).unwrap(),
            nback_runtime_observation(&config).unwrap(),
        ] {
            let json = serde_json::to_string(&observation).unwrap();
            let decoded: RuntimeCalibrationObservation = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, observation);
            assert_eq!(decoded.digest_hex().unwrap(), observation.digest_hex().unwrap());
        }

        let stroop = stroop_runtime_observation(&config).unwrap();
        assert!(validate_stroop_against_config(&stroop, &config).is_ok());
        let nback = nback_runtime_observation(&config).unwrap();
        assert!(validate_nback_against_config(&nback, &config).is_ok());
    }
}
