// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lossless exact metadata sidecar for humanoid capability observations.
//!
//! V1 continuous measurements remain unchanged. V2 wraps an already-valid V1
//! observation with bounded typed metadata for exact integers, booleans,
//! semantic tokens, and artifact/configuration references.

use crate::capability_observatory::{
    CapabilityObservationError, HumanoidCapabilityObservationV1, MeasurementProvenance,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const HUMANOID_CAPABILITY_OBSERVATION_SCHEMA_V2: &str =
    "symthaea.humanoid.capability-observation.v2";
const EXACT_METADATA_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea:humanoid-capability-exact-metadata:v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityExactMetadataValueV1 {
    Signed(i64),
    Unsigned(u64),
    Bool(bool),
    /// Short semantic/status vocabulary item. Not a free-form payload.
    Token(String),
    /// Bounded opaque artifact/configuration/calibration reference.
    Reference(String),
}

impl CapabilityExactMetadataValueV1 {
    fn validate(&self) -> Result<(), CapabilityObservationV2Error> {
        match self {
            Self::Token(value) => validate_bounded_text(
                value,
                192,
                CapabilityObservationV2Error::InvalidMetadataToken,
            ),
            Self::Reference(value) => validate_bounded_text(
                value,
                512,
                CapabilityObservationV2Error::InvalidMetadataReference,
            ),
            Self::Signed(_) | Self::Unsigned(_) | Self::Bool(_) => Ok(()),
        }
    }

    fn commit_into(&self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Signed(value) => {
                hasher.update(&[0]);
                hasher.update(&value.to_le_bytes());
            }
            Self::Unsigned(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            Self::Bool(value) => {
                hasher.update(&[2, u8::from(*value)]);
            }
            Self::Token(value) => {
                hasher.update(&[3]);
                hash_str(hasher, value);
            }
            Self::Reference(value) => {
                hasher.update(&[4]);
                hash_str(hasher, value);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityExactMetadataV1 {
    pub field_id: String,
    pub value: CapabilityExactMetadataValueV1,
    pub provenance: MeasurementProvenance,
    /// Optional exact unit when the scalar has one. Seed/status/reference fields
    /// should normally leave this unset rather than inventing a unit.
    pub unit: Option<String>,
    /// Optional schema/type identity for opaque external references.
    pub schema_id: Option<String>,
}

impl CapabilityExactMetadataV1 {
    pub fn validate(&self) -> Result<(), CapabilityObservationV2Error> {
        validate_bounded_text(
            &self.field_id,
            192,
            CapabilityObservationV2Error::InvalidMetadataFieldId,
        )?;
        self.value.validate()?;
        if let Some(unit) = &self.unit {
            validate_bounded_text(
                unit,
                64,
                CapabilityObservationV2Error::InvalidMetadataUnit,
            )?;
        }
        if let Some(schema_id) = &self.schema_id {
            validate_bounded_text(
                schema_id,
                192,
                CapabilityObservationV2Error::InvalidMetadataSchemaId,
            )?;
        }
        Ok(())
    }

    fn commit_into(&self, hasher: &mut blake3::Hasher) {
        hash_str(hasher, &self.field_id);
        hasher.update(&[provenance_tag(self.provenance)]);
        hash_optional_str(hasher, self.unit.as_deref());
        hash_optional_str(hasher, self.schema_id.as_deref());
        self.value.commit_into(hasher);
    }
}

/// Compatibility-preserving v2 envelope.
///
/// The V1 observation remains the authoritative continuous-measurement envelope.
/// `exact_metadata` adds only values that must not be coerced to floating point
/// or hidden in compound identifier strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityObservationV2 {
    pub observation: HumanoidCapabilityObservationV1,
    pub exact_metadata: Vec<CapabilityExactMetadataV1>,
}

impl HumanoidCapabilityObservationV2 {
    pub fn new(
        observation: HumanoidCapabilityObservationV1,
        exact_metadata: Vec<CapabilityExactMetadataV1>,
    ) -> Result<Self, CapabilityObservationV2Error> {
        let value = Self {
            observation,
            exact_metadata,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn from_v1(
        observation: HumanoidCapabilityObservationV1,
    ) -> Result<Self, CapabilityObservationV2Error> {
        Self::new(observation, Vec::new())
    }

    pub fn validate(&self) -> Result<(), CapabilityObservationV2Error> {
        self.observation
            .validate()
            .map_err(CapabilityObservationV2Error::InvalidV1Observation)?;

        let mut field_ids = BTreeSet::new();
        for metadata in &self.exact_metadata {
            metadata.validate()?;
            if !field_ids.insert(metadata.field_id.as_str()) {
                return Err(CapabilityObservationV2Error::DuplicateMetadataFieldId);
            }
        }
        Ok(())
    }

    /// Canonical, order-independent commitment to the exact metadata set.
    ///
    /// Vectors are accepted for simple serde compatibility, but commitment
    /// semantics sort by unique `field_id` so serialization/insertion order
    /// cannot change evidence identity.
    pub fn exact_metadata_commitment_v1(
        &self,
    ) -> Result<String, CapabilityObservationV2Error> {
        self.validate()?;
        let mut ordered: Vec<&CapabilityExactMetadataV1> = self.exact_metadata.iter().collect();
        ordered.sort_by(|left, right| left.field_id.cmp(&right.field_id));

        let mut hasher = blake3::Hasher::new();
        hasher.update(EXACT_METADATA_COMMITMENT_DOMAIN_V1);
        hash_str(&mut hasher, HUMANOID_CAPABILITY_OBSERVATION_SCHEMA_V2);
        hasher.update(&(ordered.len() as u32).to_le_bytes());
        for metadata in ordered {
            metadata.commit_into(&mut hasher);
        }
        Ok(format!(
            "humanoid-capability-exact-metadata:{}",
            hasher.finalize().to_hex()
        ))
    }

    pub fn metadata(&self, field_id: &str) -> Option<&CapabilityExactMetadataV1> {
        self.exact_metadata
            .iter()
            .find(|metadata| metadata.field_id == field_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityObservationV2Error {
    InvalidV1Observation(CapabilityObservationError),
    InvalidMetadataFieldId,
    InvalidMetadataToken,
    InvalidMetadataReference,
    InvalidMetadataUnit,
    InvalidMetadataSchemaId,
    DuplicateMetadataFieldId,
}

fn validate_bounded_text(
    value: &str,
    maximum_len: usize,
    error: CapabilityObservationV2Error,
) -> Result<(), CapabilityObservationV2Error> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > maximum_len {
        Err(error)
    } else {
        Ok(())
    }
}

fn provenance_tag(provenance: MeasurementProvenance) -> u8 {
    match provenance {
        MeasurementProvenance::Measured => 0,
        MeasurementProvenance::Derived => 1,
        MeasurementProvenance::BenchmarkNative => 2,
        MeasurementProvenance::SimulationOnly => 3,
    }
}

fn hash_optional_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_str(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_observatory::{
        BenchmarkIdentityV1, CapabilityDomain, CapabilityMeasurementV1,
        CapabilitySubjectIdentityV1, ExecutionSubstrate, RunDisposition,
    };

    fn observation() -> HumanoidCapabilityObservationV1 {
        HumanoidCapabilityObservationV1 {
            run_id: "typed-evidence-run".into(),
            domain: CapabilityDomain::Locomotion,
            substrate: ExecutionSubstrate::ExternalBenchmarkSimulation,
            subject: CapabilitySubjectIdentityV1 {
                source_head: "candidate-head".into(),
                model_or_policy_id: "policy-v1".into(),
                morphology_id: "humanoid-v1".into(),
                sensor_actuator_profile_id: "sim-v1".into(),
            },
            benchmark: BenchmarkIdentityV1::External {
                benchmark_id: "external-benchmark".into(),
                benchmark_version: "exact-revision".into(),
                task_set_id: "task-set-commitment".into(),
            },
            environment_id: "environment-v1".into(),
            authority_profile_id: "simulation-only".into(),
            evidence_profile_id: "typed-evidence-v1".into(),
            disposition: RunDisposition::Completed,
            measurements: vec![CapabilityMeasurementV1 {
                metric_id: "return".into(),
                value: 1.25,
                unit: "benchmark_return".into(),
                provenance: MeasurementProvenance::BenchmarkNative,
            }],
            failures: Vec::new(),
        }
    }

    fn field(field_id: &str, value: CapabilityExactMetadataValueV1) -> CapabilityExactMetadataV1 {
        CapabilityExactMetadataV1 {
            field_id: field_id.into(),
            value,
            provenance: MeasurementProvenance::BenchmarkNative,
            unit: None,
            schema_id: None,
        }
    }

    #[test]
    fn u64_max_round_trips_without_float_coercion() {
        let value = HumanoidCapabilityObservationV2::new(
            observation(),
            vec![field(
                "benchmark.random_seed",
                CapabilityExactMetadataValueV1::Unsigned(u64::MAX),
            )],
        )
        .unwrap();
        let encoded = serde_json::to_vec(&value).unwrap();
        let decoded: HumanoidCapabilityObservationV2 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(
            decoded.metadata("benchmark.random_seed").unwrap().value,
            CapabilityExactMetadataValueV1::Unsigned(u64::MAX)
        );
    }

    #[test]
    fn typed_boolean_is_not_numeric_surrogate() {
        let value = HumanoidCapabilityObservationV2::new(
            observation(),
            vec![field(
                "benchmark.truncated",
                CapabilityExactMetadataValueV1::Bool(true),
            )],
        )
        .unwrap();
        assert_eq!(
            value.metadata("benchmark.truncated").unwrap().value,
            CapabilityExactMetadataValueV1::Bool(true)
        );
    }

    #[test]
    fn metadata_commitment_is_independent_of_insertion_order() {
        let a = HumanoidCapabilityObservationV2::new(
            observation(),
            vec![
                field(
                    "benchmark.seed",
                    CapabilityExactMetadataValueV1::Unsigned(7),
                ),
                field(
                    "benchmark.truncated",
                    CapabilityExactMetadataValueV1::Bool(false),
                ),
            ],
        )
        .unwrap();
        let b = HumanoidCapabilityObservationV2::new(
            observation(),
            vec![
                field(
                    "benchmark.truncated",
                    CapabilityExactMetadataValueV1::Bool(false),
                ),
                field(
                    "benchmark.seed",
                    CapabilityExactMetadataValueV1::Unsigned(7),
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            a.exact_metadata_commitment_v1().unwrap(),
            b.exact_metadata_commitment_v1().unwrap()
        );
    }

    #[test]
    fn duplicate_metadata_field_ids_fail_closed() {
        let error = HumanoidCapabilityObservationV2::new(
            observation(),
            vec![
                field("duplicate", CapabilityExactMetadataValueV1::Bool(true)),
                field("duplicate", CapabilityExactMetadataValueV1::Bool(false)),
            ],
        )
        .unwrap_err();
        assert_eq!(error, CapabilityObservationV2Error::DuplicateMetadataFieldId);
    }

    #[test]
    fn oversized_reference_fails_closed() {
        let error = HumanoidCapabilityObservationV2::new(
            observation(),
            vec![field(
                "artifact",
                CapabilityExactMetadataValueV1::Reference("x".repeat(513)),
            )],
        )
        .unwrap_err();
        assert_eq!(error, CapabilityObservationV2Error::InvalidMetadataReference);
    }

    #[test]
    fn v1_migration_preserves_existing_observation_exactly() {
        let source = observation();
        let v2 = HumanoidCapabilityObservationV2::from_v1(source.clone()).unwrap();
        assert_eq!(v2.observation, source);
        assert!(v2.exact_metadata.is_empty());
    }
}
