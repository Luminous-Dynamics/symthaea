// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical physical observation types for Symthaea FIELD.
//!
//! `FieldObservationV1` is deliberately measurement-only. It cannot express
//! actuator commands or authority. Physical, simulated, and replay observations
//! share one envelope while retaining source class, clock domain, calibration,
//! uncertainty, coordinate frame, provenance, and validity semantics.
//!
//! Numeric payloads are already normalized into the declared SI unit. Unit
//! prefixes and device-native scales belong in adapters/calibration evidence,
//! never in an implicit convention inside this contract.

#![forbid(unsafe_code)]

mod canonical;
mod validation;

use serde::{Deserialize, Serialize};
use std::fmt;

pub const FIELD_OBSERVATION_SCHEMA_V1: u16 = 1;
pub const MAX_INLINE_VECTOR_VALUES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldModality {
    Acoustic,
    Optical,
    Plasma,
    Electromagnetic,
    Thermal,
    Chemical,
    Mechanical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantityKind {
    Pressure,
    Frequency,
    Wavelength,
    PhaseAngle,
    Time,
    Power,
    Energy,
    Force,
    Intensity,
    Irradiance,
    Voltage,
    Current,
    Temperature,
    Distance,
    Displacement,
    Velocity,
    Acceleration,
    AngularVelocity,
    MagneticFluxDensity,
    ElectricFieldStrength,
    MassDensity,
    NumberDensity,
    Concentration,
    SpectralIntensity,
    Dimensionless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SiUnit {
    Pascal,
    Hertz,
    Metre,
    Radian,
    Second,
    Watt,
    Joule,
    Newton,
    WattPerSquareMetre,
    Volt,
    Ampere,
    Kelvin,
    MetrePerSecond,
    MetrePerSecondSquared,
    RadianPerSecond,
    Tesla,
    VoltPerMetre,
    KilogramPerCubicMetre,
    PerCubicMetre,
    MolePerCubicMetre,
    WattPerSquareMetreHertz,
    One,
}

impl SiUnit {
    pub fn is_compatible_with(self, quantity: QuantityKind) -> bool {
        matches!(
            (quantity, self),
            (QuantityKind::Pressure, Self::Pascal)
                | (QuantityKind::Frequency, Self::Hertz)
                | (QuantityKind::Wavelength, Self::Metre)
                | (QuantityKind::PhaseAngle, Self::Radian)
                | (QuantityKind::Time, Self::Second)
                | (QuantityKind::Power, Self::Watt)
                | (QuantityKind::Energy, Self::Joule)
                | (QuantityKind::Force, Self::Newton)
                | (QuantityKind::Intensity, Self::WattPerSquareMetre)
                | (QuantityKind::Irradiance, Self::WattPerSquareMetre)
                | (QuantityKind::Voltage, Self::Volt)
                | (QuantityKind::Current, Self::Ampere)
                | (QuantityKind::Temperature, Self::Kelvin)
                | (QuantityKind::Distance, Self::Metre)
                | (QuantityKind::Displacement, Self::Metre)
                | (QuantityKind::Velocity, Self::MetrePerSecond)
                | (QuantityKind::Acceleration, Self::MetrePerSecondSquared)
                | (QuantityKind::AngularVelocity, Self::RadianPerSecond)
                | (QuantityKind::MagneticFluxDensity, Self::Tesla)
                | (QuantityKind::ElectricFieldStrength, Self::VoltPerMetre)
                | (QuantityKind::MassDensity, Self::KilogramPerCubicMetre)
                | (QuantityKind::NumberDensity, Self::PerCubicMetre)
                | (QuantityKind::Concentration, Self::MolePerCubicMetre)
                | (QuantityKind::SpectralIntensity, Self::WattPerSquareMetreHertz)
                | (QuantityKind::Dimensionless, Self::One)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceClass {
    Physical,
    Simulated,
    Replay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnavailableReason {
    NoSample,
    SensorUnavailable,
    BelowDetectionLimit,
    AboveDetectionLimit,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MeasurementPayload {
    Scalar(f64),
    /// Inline ordered components. `schema_id` defines component meaning and order.
    Vector {
        values: Vec<f64>,
        schema_id: String,
    },
    /// Reference to structured measurement bytes such as a waveform or spectrum.
    StructuredReference {
        evidence: EvidenceIdentity,
        /// Stable schema identifier defining how referenced bytes are interpreted.
        schema_id: String,
        sample_count: Option<u64>,
    },
    /// Explicit absence of a measurement. This can never be confused with zero.
    Unavailable { reason: UnavailableReason },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonotonicInstantV1 {
    pub epoch_id: String,
    pub ticks_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CaptureTimeV1 {
    /// UTC Unix time. This is a wall clock and is never monotonic.
    UnixUtc { unix_nanos: i64 },
    /// Monotonic time scoped to one explicit process/boot/device epoch.
    Monotonic { epoch_id: String, ticks_ns: u64 },
    /// Time on a replay timeline; it cannot be promoted to a physical clock.
    Replay { timeline_id: String, ticks_ns: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibrationState {
    Valid,
    NotRequired,
    Unverified,
    Expired,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationContextV1 {
    pub calibration_id: Option<String>,
    pub state: CalibrationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum UncertaintyV1 {
    Unknown,
    StandardDeviation { sigma: f64 },
    AbsoluteBound { half_width: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DigestAlgorithm {
    Sha256,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceIdentity {
    pub algorithm: DigestAlgorithm,
    pub digest_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceV1 {
    pub evidence: EvidenceIdentity,
    /// Redundant by design: disagreement with the envelope is a hard error.
    pub asserted_source_class: SourceClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationValidity {
    Nominal,
    Degraded,
    Invalid,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityFlagsV1 {
    pub clipped: bool,
    pub saturated: bool,
    pub low_snr: bool,
    pub out_of_range: bool,
}

impl QualityFlagsV1 {
    pub fn any(self) -> bool {
        self.clipped || self.saturated || self.low_snr || self.out_of_range
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldObservationV1 {
    pub schema_version: u16,
    pub observation_id: String,
    pub modality: FieldModality,
    pub quantity: QuantityKind,
    pub payload: MeasurementPayload,
    pub unit: SiUnit,
    pub source_class: SourceClass,
    pub source_id: String,
    pub coordinate_frame_id: String,
    pub capture_time: CaptureTimeV1,
    pub received_at_monotonic: MonotonicInstantV1,
    pub calibration: CalibrationContextV1,
    pub uncertainty: UncertaintyV1,
    pub provenance: ProvenanceV1,
    pub validity: ObservationValidity,
    pub quality: QualityFlagsV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    UnsupportedSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    NonFiniteMeasurement,
    EmptyVector,
    InlineVectorTooLarge { len: usize, max: usize },
    InvalidSampleCount,
    IncompatibleQuantityUnit,
    InvalidClockIdentity,
    SourceClockMismatch,
    ReceiveBeforeCapture,
    InvalidCalibrationIdentity,
    ExpiredCalibration,
    InvalidCalibration,
    CalibrationValidityMismatch,
    InvalidUncertainty,
    UnavailableUncertaintyMismatch,
    InvalidDigest,
    SourceProvenanceMismatch,
    QualityValidityMismatch,
    UnavailableNominal,
    InvalidObservation,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported FIELD schema version {version}")
            }
            Self::InvalidIdentifier(field) => write!(f, "invalid {field} identifier"),
            Self::NonFiniteMeasurement => write!(f, "measurement contains NaN or infinity"),
            Self::EmptyVector => write!(f, "measurement vector is empty"),
            Self::InlineVectorTooLarge { len, max } => {
                write!(f, "inline vector has {len} values; maximum is {max}")
            }
            Self::InvalidSampleCount => write!(f, "referenced sample count must be positive"),
            Self::IncompatibleQuantityUnit => write!(f, "quantity and SI unit are incompatible"),
            Self::InvalidClockIdentity => {
                write!(f, "clock domain identity is invalid or ambiguous")
            }
            Self::SourceClockMismatch => write!(f, "replay source class and replay clock disagree"),
            Self::ReceiveBeforeCapture => {
                write!(f, "receive timestamp precedes capture in the same monotonic epoch")
            }
            Self::InvalidCalibrationIdentity => {
                write!(f, "calibration identity/state is inconsistent")
            }
            Self::ExpiredCalibration => write!(f, "calibration is explicitly expired"),
            Self::InvalidCalibration => write!(f, "calibration is explicitly invalid"),
            Self::CalibrationValidityMismatch => {
                write!(f, "nominal validity contradicts unverified calibration")
            }
            Self::InvalidUncertainty => write!(f, "uncertainty is negative or non-finite"),
            Self::UnavailableUncertaintyMismatch => {
                write!(f, "unavailable measurement cannot carry numeric uncertainty")
            }
            Self::InvalidDigest => write!(f, "evidence digest is not canonical SHA-256 hex"),
            Self::SourceProvenanceMismatch => write!(f, "source class contradicts provenance"),
            Self::QualityValidityMismatch => {
                write!(f, "nominal validity contradicts degraded quality flags")
            }
            Self::UnavailableNominal => {
                write!(f, "unavailable measurement cannot be nominal")
            }
            Self::InvalidObservation => write!(f, "observation is explicitly invalid"),
        }
    }
}

impl std::error::Error for ValidationError {}

impl FieldObservationV1 {
    /// Validate structural and trust-boundary invariants without mutating the
    /// original observation. Staleness is intentionally a later policy check.
    pub fn validate(&self) -> Result<(), ValidationError> {
        validation::validate(self)
    }

    /// Convert into a wrapper that cannot be constructed without validation.
    pub fn into_validated(self) -> Result<ValidatedFieldObservationV1, ValidationError> {
        self.validate()?;
        Ok(ValidatedFieldObservationV1(self))
    }

    /// Deterministic, domain-separated bytes suitable as input to a
    /// cryptographic hash. Invalid observations cannot obtain canonical bytes.
    /// This representation is intentionally independent of serde JSON.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ValidationError> {
        self.validate()?;
        Ok(canonical::canonical_bytes(self))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedFieldObservationV1(FieldObservationV1);

impl ValidatedFieldObservationV1 {
    pub fn as_observation(&self) -> &FieldObservationV1 {
        &self.0
    }

    pub fn into_inner(self) -> FieldObservationV1 {
        self.0
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical::canonical_bytes(&self.0)
    }
}

impl AsRef<FieldObservationV1> for ValidatedFieldObservationV1 {
    fn as_ref(&self) -> &FieldObservationV1 {
        self.as_observation()
    }
}

impl TryFrom<FieldObservationV1> for ValidatedFieldObservationV1 {
    type Error = ValidationError;

    fn try_from(value: FieldObservationV1) -> Result<Self, Self::Error> {
        value.into_validated()
    }
}

#[cfg(test)]
mod tests;
