// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-preserving observation records for scientific sensing.
//!
//! This schema is additive to the lightweight HalSensorAdapter Vec<f32>
//! interface. Hardware adapters can enrich samples into an ObservationEnvelope
//! at the acquisition boundary. HDC vectors and learned representations are
//! derived features; they must not replace source measurements.
//!
//! Validation here checks structure only. It does not prove calibration,
//! accuracy, instrument identity, or scientific validity.

use serde::{Deserialize, Serialize};

/// Current serialized observation schema version.
pub const OBSERVATION_SCHEMA_VERSION: u16 = 1;

/// One scalar measurement with enough context to audit and replay its use.
///
/// Prefer UCUM-compatible unit codes (e.g. Cel, Pa, V, or 1 for dimensionless
/// values). Monotonic time is relative to a stream/session and is not directly
/// comparable across sessions. Wall-clock time is optional because some
/// embedded devices have no trustworthy clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationEnvelope {
    /// Schema version, currently OBSERVATION_SCHEMA_VERSION.
    pub schema_version: u16,
    /// Stable identifier for this record.
    pub observation_id: String,
    /// Stable identifier for the physical sensor instance.
    pub sensor_id: String,
    /// Identifier for the instrument or instrument assembly.
    pub instrument_id: String,
    /// Named instrument channel, for example ambient.
    pub channel: String,
    /// Semantic property measured, for example air.temperature.
    pub observed_property: String,
    /// Converted measurement value in unit.
    pub value: f64,
    /// Unit for value; prefer UCUM-compatible codes.
    pub unit: String,
    /// UTC Unix timestamp in nanoseconds, if trustworthy and available.
    pub captured_at_unix_ns: Option<u64>,
    /// Monotonic timestamp in nanoseconds relative to this acquisition session.
    pub monotonic_time_ns: u64,
    /// Increasing sequence number within this sensor stream.
    pub sequence: u64,
    /// Standard measurement uncertainty in the same unit as value.
    ///
    /// None means not supplied, not zero uncertainty.
    pub standard_uncertainty: Option<f64>,
    /// Declared calibration state or reference, which must be verified by consumers.
    pub calibration: CalibrationState,
    /// Quality warnings are retained rather than silently erased.
    pub quality_flags: Vec<ObservationQualityFlag>,
    /// Optional SHA-256 digest of the original raw sample bytes.
    ///
    /// A digest supports integrity checks only when compared with a trusted digest;
    /// it does not authenticate the instrument or prove provenance.
    pub raw_sample_sha256: Option<String>,
    /// Sensor firmware version, when available.
    pub firmware_version: Option<String>,
}

/// Calibration evidence declared by the acquisition adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CalibrationState {
    /// Calibration state was not supplied or could not be established.
    Unknown,
    /// Instrument is explicitly known to be uncalibrated.
    Uncalibrated,
    /// A record is referenced; this is not itself proof of traceability.
    Referenced {
        calibration_record_id: String,
        reference_standard_id: String,
        valid_until_unix_ns: Option<u64>,
    },
}

/// Non-exclusive quality flags attached to an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationQualityFlag {
    /// Signal reached a measurement limit.
    Saturated,
    /// Value is outside the adapter's declared operating range.
    OutOfRange,
    /// Sample is older than the adapter's freshness limit.
    Stale,
    /// Read, self-test, or plausibility check reported a sensor fault.
    SensorFault,
    /// Wall-clock synchronization is absent or unreliable.
    ClockUnsynchronized,
    /// One or more samples were dropped in the acquisition stream.
    DroppedSamples,
    /// Operator marked the observation as suspect.
    OperatorMarkedSuspect,
}

/// Structural validation failures for an observation record.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObservationValidationError {
    /// Reader does not support this schema version.
    #[error("unsupported observation schema version: {found}")]
    UnsupportedSchemaVersion { found: u16 },
    /// Required identifier or semantic label is empty.
    #[error("observation field must not be empty: {field}")]
    EmptyField { field: &'static str },
    /// Measurement value is NaN or infinite.
    #[error("observation value must be finite")]
    NonFiniteValue,
    /// Uncertainty is negative, NaN, or infinite.
    #[error("standard uncertainty must be finite and non-negative")]
    InvalidUncertainty,
    /// Raw-sample SHA-256 digest is malformed.
    #[error("raw_sample_sha256 must contain exactly 64 hexadecimal characters")]
    InvalidRawSampleDigest,
    /// A declared calibration reference has an empty identifier.
    #[error("calibration reference field must not be empty: {field}")]
    EmptyCalibrationReference { field: &'static str },
}

impl ObservationEnvelope {
    /// Check schema and structural validity without making scientific claims.
    ///
    /// Quality flags, absent uncertainty, and unknown calibration do not make a
    /// record malformed. Downstream consumers must judge fitness for their task.
    pub fn validate_structure(&self) -> Result<(), ObservationValidationError> {
        if self.schema_version != OBSERVATION_SCHEMA_VERSION {
            return Err(ObservationValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        for (field, value) in [
            ("observation_id", self.observation_id.as_str()),
            ("sensor_id", self.sensor_id.as_str()),
            ("instrument_id", self.instrument_id.as_str()),
            ("channel", self.channel.as_str()),
            ("observed_property", self.observed_property.as_str()),
            ("unit", self.unit.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ObservationValidationError::EmptyField { field });
            }
        }

        if !self.value.is_finite() {
            return Err(ObservationValidationError::NonFiniteValue);
        }

        if let Some(uncertainty) = self.standard_uncertainty {
            if !uncertainty.is_finite() || uncertainty < 0.0 {
                return Err(ObservationValidationError::InvalidUncertainty);
            }
        }

        if let Some(digest) = self.raw_sample_sha256.as_deref() {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(ObservationValidationError::InvalidRawSampleDigest);
            }
        }

        if let CalibrationState::Referenced {
            calibration_record_id,
            reference_standard_id,
            ..
        } = &self.calibration
        {
            if calibration_record_id.trim().is_empty() {
                return Err(ObservationValidationError::EmptyCalibrationReference {
                    field: "calibration_record_id",
                });
            }
            if reference_standard_id.trim().is_empty() {
                return Err(ObservationValidationError::EmptyCalibrationReference {
                    field: "reference_standard_id",
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ObservationEnvelope {
        ObservationEnvelope {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            observation_id: "run-01:thermometer-01:42".into(),
            sensor_id: "thermometer-01".into(),
            instrument_id: "sense-node-01".into(),
            channel: "ambient".into(),
            observed_property: "air.temperature".into(),
            value: 21.75,
            unit: "Cel".into(),
            captured_at_unix_ns: Some(1_800_000_000_000_000_000),
            monotonic_time_ns: 5_250_000_000,
            sequence: 42,
            standard_uncertainty: Some(0.2),
            calibration: CalibrationState::Unknown,
            quality_flags: Vec::new(),
            raw_sample_sha256: Some("a".repeat(64)),
            firmware_version: Some("0.4.1".into()),
        }
    }

    #[test]
    fn valid_observation_validates_and_round_trips() {
        let observation = sample();
        observation.validate_structure().unwrap();
        let encoded = serde_json::to_string(&observation).unwrap();
        let decoded: ObservationEnvelope = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, observation);
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let mut observation = sample();
        observation.schema_version = OBSERVATION_SCHEMA_VERSION + 1;
        assert_eq!(
            observation.validate_structure(),
            Err(ObservationValidationError::UnsupportedSchemaVersion {
                found: OBSERVATION_SCHEMA_VERSION + 1
            })
        );
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut observation = sample();
        observation.value = f64::NAN;
        assert_eq!(
            observation.validate_structure(),
            Err(ObservationValidationError::NonFiniteValue)
        );
    }

    #[test]
    fn negative_or_non_finite_uncertainty_is_rejected() {
        let mut observation = sample();
        for invalid in [-0.01, f64::INFINITY, f64::NAN] {
            observation.standard_uncertainty = Some(invalid);
            assert_eq!(
                observation.validate_structure(),
                Err(ObservationValidationError::InvalidUncertainty)
            );
        }
    }

    #[test]
    fn malformed_raw_digest_is_rejected() {
        let mut observation = sample();
        observation.raw_sample_sha256 = Some("not-a-sha256".into());
        assert_eq!(
            observation.validate_structure(),
            Err(ObservationValidationError::InvalidRawSampleDigest)
        );
    }

    #[test]
    fn empty_units_are_rejected() {
        let mut observation = sample();
        observation.unit = "  ".into();
        assert_eq!(
            observation.validate_structure(),
            Err(ObservationValidationError::EmptyField { field: "unit" })
        );
    }

    #[test]
    fn absent_uncertainty_and_unknown_calibration_are_explicitly_allowed() {
        let mut observation = sample();
        observation.standard_uncertainty = None;
        observation.calibration = CalibrationState::Unknown;
        assert!(observation.validate_structure().is_ok());
    }

    #[test]
    fn empty_calibration_reference_is_rejected() {
        let mut observation = sample();
        observation.calibration = CalibrationState::Referenced {
            calibration_record_id: " ".into(),
            reference_standard_id: "reference-01".into(),
            valid_until_unix_ns: None,
        };
        assert_eq!(
            observation.validate_structure(),
            Err(ObservationValidationError::EmptyCalibrationReference {
                field: "calibration_record_id"
            })
        );

        observation.calibration = CalibrationState::Referenced {
            calibration_record_id: "calibration-record-01".into(),
            reference_standard_id: " ".into(),
            valid_until_unix_ns: None,
        };
        assert_eq!(
            observation.validate_structure(),
            Err(ObservationValidationError::EmptyCalibrationReference {
                field: "reference_standard_id"
            })
        );
    }

    #[test]
    fn quality_flags_are_preserved_not_treated_as_schema_errors() {
        let mut observation = sample();
        observation.quality_flags = vec![
            ObservationQualityFlag::Saturated,
            ObservationQualityFlag::ClockUnsynchronized,
        ];
        assert!(observation.validate_structure().is_ok());
        assert_eq!(observation.quality_flags.len(), 2);
    }
}
