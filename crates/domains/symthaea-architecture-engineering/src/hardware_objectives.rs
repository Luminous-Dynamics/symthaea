//! Typed, provenance-bound objective registry for hardware-aware architecture search.
//!
//! The registry separates the unit and normalization of an objective value from the unit and
//! model used to account for the cost of measuring it. It also distinguishes measured values,
//! structural/device infeasibility, not-yet-measured objectives, and failed measurements.
//! Missing or failed latency is never encoded as zero latency, and an infeasible architecture
//! is never assigned an invented energy or latency value.
//!
//! This module is intended to be the import/measurement contract around Pareto-regret VoC.
//! It does not acquire datasets or claim that a source artifact is truthful; it checks exact
//! artifact digests and semantic compatibility once the acquisition layer supplies them.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REGISTRY_SCHEMA: &str = "symthaea.nas.objective-registry";
const REGISTRY_VERSION: &str = "0.1.0";
const DIGEST_DOMAIN: &[u8] = b"symthaea.nas.objective-registry.v0.1";
const SHA256_HEX_LEN: usize = 64;
const FLOAT_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveSource {
    StaticArchitectureMetadata,
    TrainedEvaluation,
    ImportedBenchmark,
    ImportedPhysicalDeviceMeasurement,
    PhysicalDeviceMeasurement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfileManifest {
    pub profile_id: String,
    pub device_model: String,
    pub board_id: String,
    pub board_revision: String,
    pub firmware_sha256: String,
    pub runtime_sha256: String,
    pub compiler_toolchain_sha256: String,
    pub quantization_spec_sha256: String,
    pub measurement_protocol_sha256: String,
    pub instrument_model: String,
    pub instrument_id: String,
    pub sampling_frequency_hz: u64,
    /// Empty while constructing; set to compute_hardware_profile_sha256 before use.
    pub profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveNormalization {
    /// Fixed lower bound from a pre-registered benchmark scale.
    pub raw_min: f64,
    /// Fixed upper bound from a pre-registered benchmark scale.
    pub raw_max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveDefinition {
    pub objective_id: String,
    /// Unit of the raw objective value, e.g. milliseconds per inference or KiB.
    pub value_unit: String,
    pub direction: ObjectiveDirection,
    pub source: ObjectiveSource,
    pub normalization: ObjectiveNormalization,
    /// SHA-256 of the exact evaluator or acquisition protocol.
    pub measurement_protocol_sha256: String,
    /// Required for PhysicalDeviceMeasurement and imported hardware objectives.
    pub hardware_profile_sha256: Option<String>,
    /// Cost of acquiring one value, not the objective value itself.
    pub measurement_cost_unit: String,
    /// SHA-256 of the model that maps the measurement action to cost units.
    pub measurement_cost_model_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveRegistry {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    /// Unique objective IDs. Order is non-semantic and canonicalized for the registry digest.
    pub objectives: Vec<ObjectiveDefinition>,
    /// Exact hardware profiles referenced by physical or imported device measurements.
    pub hardware_profiles: Vec<HardwareProfileManifest>,
    /// Empty while constructing; set to compute_objective_registry_sha256 before use.
    pub registry_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementCost {
    pub resource_unit: String,
    pub units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum MeasurementOutcome {
    Measured {
        raw_value_bits: u64,
        replicate_id: String,
        sample_count: usize,
        source_artifact_sha256: String,
        cost: MeasurementCost,
    },
    /// Explicit infeasibility (e.g. flash budget exceeded), not a missing measurement.
    Infeasible {
        reason_code: String,
        source_artifact_sha256: String,
    },
    /// No measurement has been attempted or supplied.
    NotMeasured {
        reason_code: String,
    },
    /// A measurement was attempted but failed. Retryability is explicit.
    Failed {
        reason_code: String,
        retryable: bool,
        evidence_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveObservation {
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub objective_id: String,
    pub registry_sha256: String,
    pub measurement_protocol_sha256: String,
    pub hardware_profile_sha256: Option<String>,
    pub outcome: MeasurementOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScoredMeasurement {
    Measured {
        raw_value: f64,
        normalized_utility: f64,
        value_unit: String,
        acquisition_cost: MeasurementCost,
        replicate_id: String,
        sample_count: usize,
        source_artifact_sha256: String,
    },
    Infeasible {
        reason_code: String,
        source_artifact_sha256: String,
    },
    NotMeasured {
        reason_code: String,
    },
    Failed {
        reason_code: String,
        retryable: bool,
        evidence_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveRegistryError {
    pub field: String,
    pub message: String,
}

impl ObjectiveRegistryError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ObjectiveRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for ObjectiveRegistryError {}

/// Computes a stable digest after sorting objectives by ID and clearing the digest field.
pub fn compute_objective_registry_sha256(
    registry: &ObjectiveRegistry,
) -> Result<String, ObjectiveRegistryError> {
    let mut canonical = registry.clone();
    canonical.objectives.sort_by(|left, right| left.objective_id.cmp(&right.objective_id));
    canonical.hardware_profiles.sort_by(|left, right| left.profile_sha256.cmp(&right.profile_sha256));
    canonical.registry_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ObjectiveRegistryError::new("registry", error.to_string()))?;
    Ok(digest_bytes(DIGEST_DOMAIN, &bytes))
}

/// Hashes all hardware identity, software stack and measurement apparatus fields.
pub fn compute_hardware_profile_sha256(
    profile: &HardwareProfileManifest,
) -> Result<String, ObjectiveRegistryError> {
    let mut canonical = profile.clone();
    canonical.profile_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ObjectiveRegistryError::new("hardware_profile", error.to_string()))?;
    Ok(digest_bytes(b"symthaea.nas.hardware-profile.v0.1", &bytes))
}

pub fn validate_hardware_profile(
    profile: &HardwareProfileManifest,
) -> Result<(), ObjectiveRegistryError> {
    for (field, value) in [
        ("profile_id", profile.profile_id.as_str()),
        ("device_model", profile.device_model.as_str()),
        ("board_id", profile.board_id.as_str()),
        ("board_revision", profile.board_revision.as_str()),
        ("instrument_model", profile.instrument_model.as_str()),
        ("instrument_id", profile.instrument_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(ObjectiveRegistryError::new(field, "must be non-empty"));
        }
    }
    for (field, value) in [
        ("firmware_sha256", profile.firmware_sha256.as_str()),
        ("runtime_sha256", profile.runtime_sha256.as_str()),
        ("compiler_toolchain_sha256", profile.compiler_toolchain_sha256.as_str()),
        ("quantization_spec_sha256", profile.quantization_spec_sha256.as_str()),
        ("measurement_protocol_sha256", profile.measurement_protocol_sha256.as_str()),
        ("profile_sha256", profile.profile_sha256.as_str()),
    ] {
        validate_digest(field, value)?;
    }
    if profile.sampling_frequency_hz == 0 {
        return Err(ObjectiveRegistryError::new(
            "sampling_frequency_hz",
            "must be positive",
        ));
    }
    if compute_hardware_profile_sha256(profile)? != profile.profile_sha256 {
        return Err(ObjectiveRegistryError::new(
            "profile_sha256",
            "does not match the canonical hardware profile",
        ));
    }
    Ok(())
}

pub fn validate_objective_registry(
    registry: &ObjectiveRegistry,
) -> Result<(), ObjectiveRegistryError> {
    if registry.schema != REGISTRY_SCHEMA {
        return Err(ObjectiveRegistryError::new(
            "schema",
            format!("expected {REGISTRY_SCHEMA}, got {:?}", registry.schema),
        ));
    }
    if registry.version != REGISTRY_VERSION {
        return Err(ObjectiveRegistryError::new(
            "version",
            format!("expected {REGISTRY_VERSION}, got {:?}", registry.version),
        ));
    }
    if registry.benchmark_id.trim().is_empty() {
        return Err(ObjectiveRegistryError::new(
            "benchmark_id",
            "must be non-empty",
        ));
    }
    if registry.objectives.is_empty() {
        return Err(ObjectiveRegistryError::new(
            "objectives",
            "at least one objective is required",
        ));
    }
    if !is_sha256(&registry.registry_sha256) {
        return Err(ObjectiveRegistryError::new(
            "registry_sha256",
            "must be a 64-character lowercase hexadecimal SHA-256 digest",
        ));
    }

    let mut profile_ids = HashSet::with_capacity(registry.hardware_profiles.len());
    let mut profile_digests = HashSet::with_capacity(registry.hardware_profiles.len());
    for (index, profile) in registry.hardware_profiles.iter().enumerate() {
        validate_hardware_profile(profile).map_err(|error| {
            ObjectiveRegistryError::new(
                format!("hardware_profiles[{index}].{}", error.field),
                error.message,
            )
        })?;
        if !profile_ids.insert(profile.profile_id.as_str())
            || !profile_digests.insert(profile.profile_sha256.as_str())
        {
            return Err(ObjectiveRegistryError::new(
                format!("hardware_profiles[{index}]"),
                "profile IDs and profile digests must be unique",
            ));
        }
    }

    let mut objective_ids = HashSet::with_capacity(registry.objectives.len());
    for (index, objective) in registry.objectives.iter().enumerate() {
        let field = format!("objectives[{index}]");
        if objective.objective_id.trim().is_empty()
            || !objective_ids.insert(objective.objective_id.as_str())
        {
            return Err(ObjectiveRegistryError::new(
                format!("{field}.objective_id"),
                "objective IDs must be non-empty and unique",
            ));
        }
        if objective.value_unit.trim().is_empty() {
            return Err(ObjectiveRegistryError::new(
                format!("{field}.value_unit"),
                "objective value unit must be non-empty",
            ));
        }
        if objective.measurement_cost_unit.trim().is_empty() {
            return Err(ObjectiveRegistryError::new(
                format!("{field}.measurement_cost_unit"),
                "measurement cost unit must be non-empty",
            ));
        }
        if !objective.normalization.raw_min.is_finite()
            || !objective.normalization.raw_max.is_finite()
            || objective.normalization.raw_max <= objective.normalization.raw_min
        {
            return Err(ObjectiveRegistryError::new(
                format!("{field}.normalization"),
                "normalization bounds must be finite and strictly increasing",
            ));
        }
        validate_digest(
            &format!("{field}.measurement_protocol_sha256"),
            &objective.measurement_protocol_sha256,
        )?;
        validate_digest(
            &format!("{field}.measurement_cost_model_sha256"),
            &objective.measurement_cost_model_sha256,
        )?;

        match objective.source {
            ObjectiveSource::PhysicalDeviceMeasurement
            | ObjectiveSource::ImportedPhysicalDeviceMeasurement => {
                let profile = objective.hardware_profile_sha256.as_deref().ok_or_else(|| {
                    ObjectiveRegistryError::new(
                        format!("{field}.hardware_profile_sha256"),
                        "physical-device objectives require an exact hardware profile digest",
                    )
                })?;
                validate_digest(&format!("{field}.hardware_profile_sha256"), profile)?;
                let profile_manifest = registry
                    .hardware_profiles
                    .iter()
                    .find(|candidate| candidate.profile_sha256 == profile)
                    .ok_or_else(|| {
                        ObjectiveRegistryError::new(
                            format!("{field}.hardware_profile_sha256"),
                            "referenced hardware profile manifest is absent from the registry",
                        )
                    })?;
                if profile_manifest.measurement_protocol_sha256
                    != objective.measurement_protocol_sha256
                {
                    return Err(ObjectiveRegistryError::new(
                        format!("{field}.measurement_protocol_sha256"),
                        "objective protocol digest must match the referenced hardware profile",
                    ));
                }
            }
            ObjectiveSource::ImportedBenchmark => {
                if let Some(profile) = objective.hardware_profile_sha256.as_deref() {
                    validate_digest(&format!("{field}.hardware_profile_sha256"), profile)?;
                    let profile_manifest = registry
                        .hardware_profiles
                        .iter()
                        .find(|candidate| candidate.profile_sha256 == profile)
                        .ok_or_else(|| {
                            ObjectiveRegistryError::new(
                                format!("{field}.hardware_profile_sha256"),
                                "referenced imported hardware profile is absent from the registry",
                            )
                        })?;
                    if profile_manifest.measurement_protocol_sha256
                        != objective.measurement_protocol_sha256
                    {
                        return Err(ObjectiveRegistryError::new(
                            format!("{field}.measurement_protocol_sha256"),
                            "objective protocol digest must match the referenced hardware profile",
                        ));
                    }
                }
            }
            ObjectiveSource::StaticArchitectureMetadata | ObjectiveSource::TrainedEvaluation => {
                if objective.hardware_profile_sha256.is_some() {
                    return Err(ObjectiveRegistryError::new(
                        format!("{field}.hardware_profile_sha256"),
                        "this source class does not accept a device profile in v0.1",
                    ));
                }
            }
        }
    }

    if compute_objective_registry_sha256(registry)? != registry.registry_sha256 {
        return Err(ObjectiveRegistryError::new(
            "registry_sha256",
            "does not match the canonical objective registry",
        ));
    }
    Ok(())
}

/// Convert a raw value to normalized maximize-is-better utility.
///
/// Values outside the frozen normalization range are rejected rather than clipped. Silent
/// clipping would hide distribution shift and distort Pareto geometry.
pub fn normalize_objective_value(
    definition: &ObjectiveDefinition,
    raw_value: f64,
) -> Result<f64, ObjectiveRegistryError> {
    if !raw_value.is_finite() {
        return Err(ObjectiveRegistryError::new(
            "raw_value",
            "must be finite",
        ));
    }
    let minimum = definition.normalization.raw_min;
    let maximum = definition.normalization.raw_max;
    if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
        return Err(ObjectiveRegistryError::new(
            "normalization",
            "bounds must be finite and strictly increasing",
        ));
    }
    if raw_value < minimum - FLOAT_TOLERANCE || raw_value > maximum + FLOAT_TOLERANCE {
        return Err(ObjectiveRegistryError::new(
            "raw_value",
            "falls outside the frozen normalization range; do not silently clip",
        ));
    }
    let fraction = ((raw_value - minimum) / (maximum - minimum)).clamp(0.0, 1.0);
    Ok(match definition.direction {
        ObjectiveDirection::Maximize => fraction,
        ObjectiveDirection::Minimize => 1.0 - fraction,
    })
}

/// Validate an observation against the registry and return a typed scoring outcome.
///
/// This function never invents a numeric substitute for Infeasible, NotMeasured or Failed.
pub fn score_objective_observation(
    registry: &ObjectiveRegistry,
    observation: &ObjectiveObservation,
) -> Result<ScoredMeasurement, ObjectiveRegistryError> {
    validate_objective_registry(registry)?;
    if observation.candidate_id.trim().is_empty() {
        return Err(ObjectiveRegistryError::new(
            "candidate_id",
            "must be non-empty",
        ));
    }
    if !is_sha256(&observation.candidate_config_sha256) {
        return Err(ObjectiveRegistryError::new(
            "candidate_config_sha256",
            "must be a 64-character lowercase hexadecimal SHA-256 digest",
        ));
    }
    if observation.registry_sha256 != registry.registry_sha256 {
        return Err(ObjectiveRegistryError::new(
            "registry_sha256",
            "observation is bound to a different objective registry",
        ));
    }
    let definition = registry
        .objectives
        .iter()
        .find(|objective| objective.objective_id == observation.objective_id)
        .ok_or_else(|| {
            ObjectiveRegistryError::new(
                "objective_id",
                format!("unknown objective {:?}", observation.objective_id),
            )
        })?;

    if observation.measurement_protocol_sha256 != definition.measurement_protocol_sha256 {
        return Err(ObjectiveRegistryError::new(
            "measurement_protocol_sha256",
            "observation protocol does not match the frozen objective definition",
        ));
    }
    if observation.hardware_profile_sha256 != definition.hardware_profile_sha256 {
        return Err(ObjectiveRegistryError::new(
            "hardware_profile_sha256",
            "observation device profile does not match the objective definition",
        ));
    }

    match &observation.outcome {
        MeasurementOutcome::Measured {
            raw_value_bits,
            replicate_id,
            sample_count,
            source_artifact_sha256,
            cost,
        } => {
            let raw_value = f64::from_bits(*raw_value_bits);
            let normalized_utility = normalize_objective_value(definition, raw_value)?;
            if replicate_id.trim().is_empty() {
                return Err(ObjectiveRegistryError::new(
                    "replicate_id",
                    "measured observations require a non-empty replicate ID",
                ));
            }
            if *sample_count == 0 {
                return Err(ObjectiveRegistryError::new(
                    "sample_count",
                    "measured observations require at least one sample",
                ));
            }
            validate_digest("source_artifact_sha256", source_artifact_sha256)?;
            if cost.resource_unit != definition.measurement_cost_unit {
                return Err(ObjectiveRegistryError::new(
                    "cost.resource_unit",
                    "acquisition-cost unit does not match the frozen objective definition",
                ));
            }
            if cost.units == 0 {
                return Err(ObjectiveRegistryError::new(
                    "cost.units",
                    "measured objective acquisition cost must be positive",
                ));
            }
            Ok(ScoredMeasurement::Measured {
                raw_value,
                normalized_utility,
                value_unit: definition.value_unit.clone(),
                acquisition_cost: cost.clone(),
                replicate_id: replicate_id.clone(),
                sample_count: *sample_count,
                source_artifact_sha256: source_artifact_sha256.clone(),
            })
        }
        MeasurementOutcome::Infeasible {
            reason_code,
            source_artifact_sha256,
        } => {
            if reason_code.trim().is_empty() {
                return Err(ObjectiveRegistryError::new(
                    "reason_code",
                    "infeasibility requires a reason code",
                ));
            }
            validate_digest("source_artifact_sha256", source_artifact_sha256)?;
            Ok(ScoredMeasurement::Infeasible {
                reason_code: reason_code.clone(),
                source_artifact_sha256: source_artifact_sha256.clone(),
            })
        }
        MeasurementOutcome::NotMeasured { reason_code } => {
            if reason_code.trim().is_empty() {
                return Err(ObjectiveRegistryError::new(
                    "reason_code",
                    "not-measured records require a reason code",
                ));
            }
            Ok(ScoredMeasurement::NotMeasured {
                reason_code: reason_code.clone(),
            })
        }
        MeasurementOutcome::Failed {
            reason_code,
            retryable,
            evidence_sha256,
        } => {
            if reason_code.trim().is_empty() {
                return Err(ObjectiveRegistryError::new(
                    "reason_code",
                    "failed measurements require a reason code",
                ));
            }
            validate_digest("evidence_sha256", evidence_sha256)?;
            Ok(ScoredMeasurement::Failed {
                reason_code: reason_code.clone(),
                retryable: *retryable,
                evidence_sha256: evidence_sha256.clone(),
            })
        }
    }
}

fn validate_digest(field: &str, value: &str) -> Result<(), ObjectiveRegistryError> {
    if !is_sha256(value) {
        return Err(ObjectiveRegistryError::new(
            field,
            "must be a 64-character lowercase hexadecimal SHA-256 digest",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == SHA256_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn objective(
        id: &str,
        direction: ObjectiveDirection,
        source: ObjectiveSource,
        hardware_profile_sha256: Option<String>,
    ) -> ObjectiveDefinition {
        ObjectiveDefinition {
            objective_id: id.into(),
            value_unit: "ms".into(),
            direction,
            source,
            normalization: ObjectiveNormalization {
                raw_min: 0.0,
                raw_max: 100.0,
            },
            measurement_protocol_sha256: sha('a'),
            hardware_profile_sha256,
            measurement_cost_unit: "measurement-queries".into(),
            measurement_cost_model_sha256: sha('b'),
        }
    }

    fn hardware_profile() -> HardwareProfileManifest {
        let mut profile = HardwareProfileManifest {
            profile_id: "nrf5340-board-01".into(),
            device_model: "Nordic nRF5340".into(),
            board_id: "board-01".into(),
            board_revision: "DK-rev-2".into(),
            firmware_sha256: sha('1'),
            runtime_sha256: sha('2'),
            compiler_toolchain_sha256: sha('3'),
            quantization_spec_sha256: sha('4'),
            measurement_protocol_sha256: sha('a'),
            instrument_model: "Nordic Power Profiler Kit II".into(),
            instrument_id: "ppk2-01".into(),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = compute_hardware_profile_sha256(&profile).unwrap();
        profile
    }

    fn registry() -> ObjectiveRegistry {
        let profile = hardware_profile();
        let mut value = ObjectiveRegistry {
            schema: REGISTRY_SCHEMA.into(),
            version: REGISTRY_VERSION.into(),
            benchmark_id: "nats-bench-mcu-v2-nrf5340".into(),
            objectives: vec![
                objective(
                    "latency",
                    ObjectiveDirection::Minimize,
                    ObjectiveSource::PhysicalDeviceMeasurement,
                    Some(profile.profile_sha256.clone()),
                ),
                objective(
                    "accuracy",
                    ObjectiveDirection::Maximize,
                    ObjectiveSource::TrainedEvaluation,
                    None,
                ),
            ],
            hardware_profiles: vec![profile],
            registry_sha256: String::new(),
        };
        value.registry_sha256 = compute_objective_registry_sha256(&value).unwrap();
        value
    }

    fn observation(outcome: MeasurementOutcome) -> ObjectiveObservation {
        let registry = registry();
        ObjectiveObservation {
            candidate_id: "arch-42".into(),
            candidate_config_sha256: sha('d'),
            objective_id: "latency".into(),
            registry_sha256: registry.registry_sha256,
            measurement_protocol_sha256: sha('a'),
            hardware_profile_sha256: Some(hardware_profile().profile_sha256),
            outcome,
        }
    }

    fn measured(raw_value: f64) -> MeasurementOutcome {
        MeasurementOutcome::Measured {
            raw_value_bits: raw_value.to_bits(),
            replicate_id: "board-1-run-1".into(),
            sample_count: 30,
            source_artifact_sha256: sha('e'),
            cost: MeasurementCost {
                resource_unit: "measurement-queries".into(),
                units: 1,
            },
        }
    }

    #[test]
    fn hardware_profile_digest_binds_software_and_instrumentation() {
        let mut profile = HardwareProfileManifest {
            profile_id: "nrf5340-board-01".into(),
            device_model: "Nordic nRF5340".into(),
            board_id: "board-01".into(),
            board_revision: "DK-rev-2".into(),
            firmware_sha256: sha('1'),
            runtime_sha256: sha('2'),
            compiler_toolchain_sha256: sha('3'),
            quantization_spec_sha256: sha('4'),
            measurement_protocol_sha256: sha('5'),
            instrument_model: "Nordic Power Profiler Kit II".into(),
            instrument_id: "ppk2-01".into(),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = compute_hardware_profile_sha256(&profile).unwrap();
        validate_hardware_profile(&profile).unwrap();

        let mut changed = profile.clone();
        changed.compiler_toolchain_sha256 = sha('6');
        assert_ne!(
            profile.profile_sha256,
            compute_hardware_profile_sha256(&changed).unwrap()
        );
    }

    #[test]
    fn imported_physical_objective_requires_profile_identity() {
        let mut value = registry();
        value.objectives[0].source = ObjectiveSource::ImportedPhysicalDeviceMeasurement;
        value.objectives[0].hardware_profile_sha256 = None;
        value.registry_sha256 = compute_objective_registry_sha256(&value).unwrap();
        assert!(validate_objective_registry(&value).is_err());
    }

    #[test]
    fn registry_digest_is_order_invariant() {
        let first = registry();
        let mut second = first.clone();
        second.objectives.reverse();
        assert_eq!(
            first.registry_sha256,
            compute_objective_registry_sha256(&second).unwrap()
        );
    }

    #[test]
    fn registry_rejects_a_device_hash_without_its_profile_manifest() {
        let mut value = registry();
        value.hardware_profiles.clear();
        value.registry_sha256 = compute_objective_registry_sha256(&value).unwrap();
        assert!(validate_objective_registry(&value).is_err());
    }

    #[test]
    fn registry_rejects_a_changed_objective_or_cost_model() {
        let first = registry();
        let mut second = first.clone();
        second.objectives[0].measurement_cost_model_sha256 = sha('f');
        assert!(validate_objective_registry(&second).is_err());
    }

    #[test]
    fn physical_device_objective_requires_profile_identity() {
        let mut value = registry();
        value.objectives[0].hardware_profile_sha256 = None;
        value.registry_sha256 = compute_objective_registry_sha256(&value).unwrap();
        assert!(validate_objective_registry(&value).is_err());
    }

    #[test]
    fn minimize_objective_maps_low_raw_value_to_high_utility() {
        let registry = registry();
        let observation = observation(measured(20.0));
        let scored = score_objective_observation(&registry, &observation).unwrap();
        match scored {
            ScoredMeasurement::Measured {
                normalized_utility,
                ..
            } => assert!((normalized_utility - 0.8).abs() < 1e-12),
            _ => panic!("expected measured value"),
        }
    }

    #[test]
    fn maximize_objective_maps_high_raw_value_to_high_utility() {
        let registry = registry();
        let mut observation = observation(measured(80.0));
        observation.objective_id = "accuracy".into();
        observation.hardware_profile_sha256 = None;
        let scored = score_objective_observation(&registry, &observation).unwrap();
        match scored {
            ScoredMeasurement::Measured {
                normalized_utility,
                ..
            } => assert!((normalized_utility - 0.8).abs() < 1e-12),
            _ => panic!("expected measured value"),
        }
    }

    #[test]
    fn infeasible_is_not_encoded_as_zero_latency_or_energy() {
        let registry = registry();
        let observation = observation(MeasurementOutcome::Infeasible {
            reason_code: "flash_budget_exceeded".into(),
            source_artifact_sha256: sha('e'),
        });
        let scored = score_objective_observation(&registry, &observation).unwrap();
        assert!(matches!(
            scored,
            ScoredMeasurement::Infeasible { .. }
        ));
    }

    #[test]
    fn not_measured_is_distinct_from_infeasible_and_failure() {
        let registry = registry();
        let not_measured = observation(MeasurementOutcome::NotMeasured {
            reason_code: "not_in_dataset".into(),
        });
        let failed = observation(MeasurementOutcome::Failed {
            reason_code: "device_timeout".into(),
            retryable: true,
            evidence_sha256: sha('e'),
        });
        assert!(matches!(
            score_objective_observation(&registry, &not_measured).unwrap(),
            ScoredMeasurement::NotMeasured { .. }
        ));
        assert!(matches!(
            score_objective_observation(&registry, &failed).unwrap(),
            ScoredMeasurement::Failed {
                retryable: true,
                ..
            }
        ));
    }

    #[test]
    fn rejects_measurement_from_a_different_device_profile() {
        let registry = registry();
        let mut observation = observation(measured(20.0));
        observation.hardware_profile_sha256 = Some(sha('f'));
        assert!(score_objective_observation(&registry, &observation).is_err());
    }

    #[test]
    fn rejects_measurement_from_a_different_protocol() {
        let registry = registry();
        let mut observation = observation(measured(20.0));
        observation.measurement_protocol_sha256 = sha('f');
        assert!(score_objective_observation(&registry, &observation).is_err());
    }

    #[test]
    fn rejects_acquisition_cost_units_that_differ_from_the_registry() {
        let registry = registry();
        let mut observation = observation(measured(20.0));
        if let MeasurementOutcome::Measured { cost, .. } = &mut observation.outcome {
            cost.resource_unit = "device-seconds".into();
        }
        assert!(score_objective_observation(&registry, &observation).is_err());
    }

    #[test]
    fn rejects_out_of_range_values_instead_of_clipping() {
        let registry = registry();
        let observation = observation(measured(120.0));
        assert!(score_objective_observation(&registry, &observation).is_err());
    }

    #[test]
    fn rejects_zero_cost_measured_observation() {
        let registry = registry();
        let mut observation = observation(measured(20.0));
        if let MeasurementOutcome::Measured { cost, .. } = &mut observation.outcome {
            cost.units = 0;
        }
        assert!(score_objective_observation(&registry, &observation).is_err());
    }
}
