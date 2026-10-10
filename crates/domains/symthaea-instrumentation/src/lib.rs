// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-aware measurement contracts for research instruments.
//!
//! This crate provides an instrument-neutral envelope suitable for ultrasound,
//! physiological sensors, and other scientific instruments. It validates
//! structure, units, sequence/time ordering, freshness, and quality flags.
//!
//! **Trust boundary:** a calibration identifier or artifact hash is not proof
//! of calibration validity or metrological traceability. Quantitative-use
//! assessment requires a caller-supplied `CalibrationEvidenceResolver` that
//! resolves and independently reviews the referenced evidence. This crate
//! validates the resolver's returned record against the measurement; it cannot
//! establish that a resolver is independent or trustworthy. That deployment
//! decision must be separately governed and tested.
//!
//! This is an engineering data contract, not a medical device, diagnosis
//! algorithm, regulatory assessment, or clinical validation claim.

use std::collections::HashMap;
use std::fmt;

/// Stable instrument and channel identifiers. Patient identity does not belong
/// in this measurement envelope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstrumentIdentity {
    instrument_id: String,
    channel_id: String,
}

impl InstrumentIdentity {
    pub fn new(
        instrument_id: impl Into<String>,
        channel_id: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let instrument_id = non_empty(instrument_id.into(), "instrument_id")?;
        let channel_id = non_empty(channel_id.into(), "channel_id")?;
        Ok(Self {
            instrument_id,
            channel_id,
        })
    }

    pub fn instrument_id(&self) -> &str {
        &self.instrument_id
    }

    pub fn channel_id(&self) -> &str {
        &self.channel_id
    }
}

/// Physical quantity named independently of the unit used to express it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantity {
    Length,
    Time,
    Frequency,
    AcousticPressure,
    Temperature,
    ElectricalPotential,
    ElectricalCurrent,
    HeartRate,
    RespiratoryRate,
    OxygenSaturation,
    Speed,
    Acceleration,
    AngularVelocity,
    SoundPressureLevel,
    Dimensionless,
}

/// Units supported by this initial measurement contract.
///
/// Extend deliberately: adding a unit requires adding its dimensional mapping
/// and tests. Values are stored in the explicitly declared unit; this crate
/// does not silently convert them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    Meter,
    Centimeter,
    Millimeter,
    Micrometer,
    Second,
    Millisecond,
    Microsecond,
    Hertz,
    Kilohertz,
    Megahertz,
    Pascal,
    Kilopascal,
    DegreeCelsius,
    Kelvin,
    Volt,
    Millivolt,
    Microvolt,
    Ampere,
    Milliampere,
    BeatsPerMinute,
    BreathsPerMinute,
    Percent,
    MeterPerSecond,
    MeterPerSecondSquared,
    RadianPerSecond,
    DecibelRe20Micropascal,
    One,
}

impl Unit {
    pub const fn quantity(self) -> Quantity {
        match self {
            Self::Meter | Self::Centimeter | Self::Millimeter | Self::Micrometer => {
                Quantity::Length
            }
            Self::Second | Self::Millisecond | Self::Microsecond => Quantity::Time,
            Self::Hertz | Self::Kilohertz | Self::Megahertz => Quantity::Frequency,
            Self::Pascal | Self::Kilopascal => Quantity::AcousticPressure,
            Self::DegreeCelsius | Self::Kelvin => Quantity::Temperature,
            Self::Volt | Self::Millivolt | Self::Microvolt => Quantity::ElectricalPotential,
            Self::Ampere | Self::Milliampere => Quantity::ElectricalCurrent,
            Self::BeatsPerMinute => Quantity::HeartRate,
            Self::BreathsPerMinute => Quantity::RespiratoryRate,
            Self::Percent => Quantity::OxygenSaturation,
            Self::MeterPerSecond => Quantity::Speed,
            Self::MeterPerSecondSquared => Quantity::Acceleration,
            Self::RadianPerSecond => Quantity::AngularVelocity,
            Self::DecibelRe20Micropascal => Quantity::SoundPressureLevel,
            Self::One => Quantity::Dimensionless,
        }
    }
}

/// Reference to an immutable evidence artifact by identifier and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactReference {
    artifact_id: String,
    sha256_hex: String,
}

impl ArtifactReference {
    pub fn new(
        artifact_id: impl Into<String>,
        sha256_hex: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let artifact_id = non_empty(artifact_id.into(), "artifact_id")?;
        let sha256_hex = sha256_hex.into();
        if sha256_hex.len() != 64 || !sha256_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(ContractError::InvalidSha256);
        }
        Ok(Self {
            artifact_id,
            sha256_hex: sha256_hex.to_ascii_lowercase(),
        })
    }

    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }
}

/// Reference supplied by the acquisition system. Its existence, contents,
/// review, and applicability are *not* established by storing this value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationReference {
    record_id: String,
    evidence: ArtifactReference,
}

impl CalibrationReference {
    pub fn new(
        record_id: impl Into<String>,
        evidence: ArtifactReference,
    ) -> Result<Self, ContractError> {
        Ok(Self {
            record_id: non_empty(record_id.into(), "calibration_record_id")?,
            evidence,
        })
    }

    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    pub fn evidence(&self) -> &ArtifactReference {
        &self.evidence
    }
}

/// Immutable raw sample or acquisition bundle reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawDataReference(ArtifactReference);

impl RawDataReference {
    pub fn new(reference: ArtifactReference) -> Self {
        Self(reference)
    }

    pub fn artifact(&self) -> &ArtifactReference {
        &self.0
    }
}

/// Explicit acquisition-quality flags. Every flag blocks the strict
/// quantitative-use gate; a future policy may distinguish safe warning classes
/// only with evidence and tests for the intended use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityFlag {
    Saturated,
    MotionArtifact,
    LeadOff,
    MissingSamples,
    AcquisitionIncomplete,
    ClockUnsynchronized,
    SelfTestFailed,
    SensorOutOfRange,
    SignalQualityUnknown,
}

/// Construction-time or record-integrity errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    EmptyIdentifier(&'static str),
    InvalidSha256,
    InvalidValidityInterval,
    CalibrationReviewMustBeDistinct,
    NonFiniteValue(&'static str),
    NegativeUncertainty(&'static str),
    QuantityUnitMismatch { quantity: Quantity, unit: Unit },
    EmptyProcessingVersion,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(f, "{field} must not be empty"),
            Self::InvalidSha256 => {
                write!(f, "SHA-256 must contain exactly 64 hexadecimal characters")
            }
            Self::InvalidValidityInterval => {
                write!(f, "validity interval must satisfy start < end")
            }
            Self::CalibrationReviewMustBeDistinct => {
                write!(f, "calibration evidence and review receipt must be distinct artifacts")
            }
            Self::NonFiniteValue(field) => write!(f, "{field} must be finite"),
            Self::NegativeUncertainty(field) => write!(f, "{field} must be non-negative"),
            Self::QuantityUnitMismatch { quantity, unit } => {
                write!(f, "unit {unit:?} is not compatible with quantity {quantity:?}")
            }
            Self::EmptyProcessingVersion => write!(f, "processing_chain_version must not be empty"),
        }
    }
}

impl std::error::Error for ContractError {}

/// A validated, immutable envelope around one scalar instrument observation.
///
/// Construction validates data shape and basic dimensional consistency. It
/// does not claim that a sensor is accurate or that supplied references are
/// authentic. Raw data and calibration references are optional for archival or
/// research capture, but the strict quantitative-use gate requires both.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementEnvelope {
    identity: InstrumentIdentity,
    sequence: u64,
    captured_at_ns: u64,
    quantity: Quantity,
    unit: Unit,
    value: f64,
    standard_uncertainty: f64,
    calibration: Option<CalibrationReference>,
    raw_data: Option<RawDataReference>,
    processing_chain_version: String,
    quality_flags: Vec<QualityFlag>,
}

#[derive(Debug, Clone)]
pub struct MeasurementInput {
    pub identity: InstrumentIdentity,
    pub sequence: u64,
    /// Monotonic acquisition timestamp in nanoseconds, from a documented clock.
    pub captured_at_ns: u64,
    pub quantity: Quantity,
    pub unit: Unit,
    pub value: f64,
    /// Standard uncertainty expressed in the same unit as `value`.
    pub standard_uncertainty: f64,
    pub calibration: Option<CalibrationReference>,
    pub raw_data: Option<RawDataReference>,
    /// Immutable version or digest identifying the transform chain applied.
    pub processing_chain_version: String,
    pub quality_flags: Vec<QualityFlag>,
}

impl MeasurementEnvelope {
    pub fn new(input: MeasurementInput) -> Result<Self, ContractError> {
        if input.quantity != input.unit.quantity() {
            return Err(ContractError::QuantityUnitMismatch {
                quantity: input.quantity,
                unit: input.unit,
            });
        }
        if !input.value.is_finite() {
            return Err(ContractError::NonFiniteValue("value"));
        }
        if !input.standard_uncertainty.is_finite() {
            return Err(ContractError::NonFiniteValue("standard_uncertainty"));
        }
        if input.standard_uncertainty < 0.0 {
            return Err(ContractError::NegativeUncertainty("standard_uncertainty"));
        }
        let processing_chain_version = input.processing_chain_version.trim().to_owned();
        if processing_chain_version.is_empty() {
            return Err(ContractError::EmptyProcessingVersion);
        }

        Ok(Self {
            identity: input.identity,
            sequence: input.sequence,
            captured_at_ns: input.captured_at_ns,
            quantity: input.quantity,
            unit: input.unit,
            value: input.value,
            standard_uncertainty: input.standard_uncertainty,
            calibration: input.calibration,
            raw_data: input.raw_data,
            processing_chain_version,
            quality_flags: input.quality_flags,
        })
    }

    pub fn identity(&self) -> &InstrumentIdentity {
        &self.identity
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn captured_at_ns(&self) -> u64 {
        self.captured_at_ns
    }

    pub fn quantity(&self) -> Quantity {
        self.quantity
    }

    pub fn unit(&self) -> Unit {
        self.unit
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn standard_uncertainty(&self) -> f64 {
        self.standard_uncertainty
    }

    pub fn quality_flags(&self) -> &[QualityFlag] {
        &self.quality_flags
    }

    pub fn calibration_reference(&self) -> Option<&CalibrationReference> {
        self.calibration.as_ref()
    }

    pub fn raw_data_reference(&self) -> Option<&RawDataReference> {
        self.raw_data.as_ref()
    }

    pub fn processing_chain_version(&self) -> &str {
        &self.processing_chain_version
    }

    /// Evaluate eligibility for a bounded quantitative computation.
    ///
    /// A successful assessment is only as trustworthy as the resolver supplied
    /// by the caller. It is not a clinical-release decision. Measurement and
    /// calibration uncertainties are kept separate; this crate does not combine
    /// them without a model of correlation and other uncertainty components.
    pub fn assess_for_quantitative_use(
        &self,
        now_ns: u64,
        policy: &MeasurementPolicy,
        resolver: &dyn CalibrationEvidenceResolver,
        stream_guard: &mut MeasurementStreamGuard,
    ) -> Result<MeasurementAssessment, AssessmentFailure> {
        stream_guard
            .observe(self)
            .map_err(AssessmentFailure::StreamOrder)?;
        if now_ns < self.captured_at_ns {
            return Err(AssessmentFailure::ClockInFuture);
        }
        let age_ns = now_ns - self.captured_at_ns;
        if age_ns > policy.max_age_ns {
            return Err(AssessmentFailure::Stale { age_ns, max_age_ns: policy.max_age_ns });
        }
        if !self.quality_flags.is_empty() {
            return Err(AssessmentFailure::QualityFlagsPresent(self.quality_flags.clone()));
        }
        let raw_data = self
            .raw_data
            .clone()
            .ok_or(AssessmentFailure::MissingRawDataReference)?;
        let reference = self
            .calibration
            .as_ref()
            .ok_or(AssessmentFailure::MissingCalibrationReference)?;
        let resolved = resolver
            .resolve(reference)
            .map_err(AssessmentFailure::CalibrationEvidenceUnresolved)?;

        if resolved.record_id != reference.record_id
            || resolved.evidence != reference.evidence
        {
            return Err(AssessmentFailure::CalibrationReferenceMismatch);
        }
        if resolved.quantity != self.quantity || resolved.unit != self.unit {
            return Err(AssessmentFailure::CalibrationUnitMismatch);
        }
        if !(resolved.valid_from_ns <= self.captured_at_ns
            && self.captured_at_ns < resolved.valid_until_ns)
        {
            return Err(AssessmentFailure::CalibrationNotValidAtCapture);
        }
        if self.standard_uncertainty > policy.max_measurement_standard_uncertainty {
            return Err(AssessmentFailure::MeasurementUncertaintyExceeded);
        }
        if resolved.standard_uncertainty > policy.max_calibration_standard_uncertainty {
            return Err(AssessmentFailure::CalibrationUncertaintyExceeded);
        }

        Ok(MeasurementAssessment {
            age_ns,
            measurement_standard_uncertainty: self.standard_uncertainty,
            calibration_standard_uncertainty: resolved.standard_uncertainty,
            calibration_record_id: resolved.record_id,
            calibration_review_receipt: resolved.review_receipt,
            raw_data,
            processing_chain_version: self.processing_chain_version.clone(),
        })
    }
}

/// An independently reviewed calibration record returned by the integration's
/// evidence resolver. The resolver is the trust boundary; this data type alone
/// does not make the record trustworthy.
///
/// Validity end is exclusive: `valid_from_ns <= captured_at_ns < valid_until_ns`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCalibration {
    record_id: String,
    evidence: ArtifactReference,
    review_receipt: ArtifactReference,
    quantity: Quantity,
    unit: Unit,
    valid_from_ns: u64,
    valid_until_ns: u64,
    standard_uncertainty: f64,
}

impl ResolvedCalibration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        record_id: impl Into<String>,
        evidence: ArtifactReference,
        review_receipt: ArtifactReference,
        quantity: Quantity,
        unit: Unit,
        valid_from_ns: u64,
        valid_until_ns: u64,
        standard_uncertainty: f64,
    ) -> Result<Self, ContractError> {
        if valid_from_ns >= valid_until_ns {
            return Err(ContractError::InvalidValidityInterval);
        }
        if evidence.artifact_id() == review_receipt.artifact_id()
            || evidence.sha256_hex() == review_receipt.sha256_hex()
        {
            return Err(ContractError::CalibrationReviewMustBeDistinct);
        }
        if quantity != unit.quantity() {
            return Err(ContractError::QuantityUnitMismatch { quantity, unit });
        }
        if !standard_uncertainty.is_finite() {
            return Err(ContractError::NonFiniteValue("calibration_standard_uncertainty"));
        }
        if standard_uncertainty < 0.0 {
            return Err(ContractError::NegativeUncertainty("calibration_standard_uncertainty"));
        }
        Ok(Self {
            record_id: non_empty(record_id.into(), "calibration_record_id")?,
            evidence,
            review_receipt,
            quantity,
            unit,
            valid_from_ns,
            valid_until_ns,
            standard_uncertainty,
        })
    }
}

/// Resolve a calibration reference from an independently governed evidence
/// store. Implementations should verify artifact bytes/digests, review receipts,
/// applicability, and traceability-chain information; a lookup by ID alone is
/// not enough. The caller must configure and test this trust boundary.
pub trait CalibrationEvidenceResolver {
    fn resolve(
        &self,
        reference: &CalibrationReference,
    ) -> Result<ResolvedCalibration, String>;
}

/// Conservative thresholds for a particular consumer and intended use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementPolicy {
    max_age_ns: u64,
    max_measurement_standard_uncertainty: f64,
    max_calibration_standard_uncertainty: f64,
}

impl MeasurementPolicy {
    pub fn new(
        max_age_ns: u64,
        max_measurement_standard_uncertainty: f64,
        max_calibration_standard_uncertainty: f64,
    ) -> Result<Self, ContractError> {
        for (name, value) in [
            ("max_measurement_standard_uncertainty", max_measurement_standard_uncertainty),
            ("max_calibration_standard_uncertainty", max_calibration_standard_uncertainty),
        ] {
            if !value.is_finite() {
                return Err(ContractError::NonFiniteValue(name));
            }
            if value < 0.0 {
                return Err(ContractError::NegativeUncertainty(name));
            }
        }
        Ok(Self {
            max_age_ns,
            max_measurement_standard_uncertainty,
            max_calibration_standard_uncertainty,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAssessment {
    pub age_ns: u64,
    pub measurement_standard_uncertainty: f64,
    pub calibration_standard_uncertainty: f64,
    pub calibration_record_id: String,
    pub calibration_review_receipt: ArtifactReference,
    pub raw_data: RawDataReference,
    pub processing_chain_version: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssessmentFailure {
    StreamOrder(StreamOrderFailure),
    ClockInFuture,
    Stale { age_ns: u64, max_age_ns: u64 },
    QualityFlagsPresent(Vec<QualityFlag>),
    MissingRawDataReference,
    MissingCalibrationReference,
    CalibrationEvidenceUnresolved(String),
    CalibrationReferenceMismatch,
    CalibrationUnitMismatch,
    CalibrationNotValidAtCapture,
    MeasurementUncertaintyExceeded,
    CalibrationUncertaintyExceeded,
}

impl fmt::Display for AssessmentFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AssessmentFailure {}

/// Per-instrument/channel guard against replayed, duplicate, or time-reversed
/// observations. A new sequence is consumed even when its timestamp regresses; the
/// stored timestamp floor remains monotonic, preventing reuse of that rejected sequence.
#[derive(Debug, Default)]
pub struct MeasurementStreamGuard {
    last: HashMap<InstrumentIdentity, (u64, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOrderFailure {
    SequenceNotIncreasing { previous: u64, received: u64 },
    TimestampMovedBackward { previous_ns: u64, received_ns: u64 },
}

impl MeasurementStreamGuard {
    pub fn observe(&mut self, measurement: &MeasurementEnvelope) -> Result<(), StreamOrderFailure> {
        let identity = measurement.identity.clone();
        if let Some((previous_sequence, previous_timestamp_ns)) =
            self.last.get(&identity).copied()
        {
            if measurement.sequence <= previous_sequence {
                return Err(StreamOrderFailure::SequenceNotIncreasing {
                    previous: previous_sequence,
                    received: measurement.sequence,
                });
            }
            if measurement.captured_at_ns < previous_timestamp_ns {
                // Consume the strictly newer sequence but preserve the high-water
                // timestamp. Retrying this sequence with a corrected timestamp must
                // not turn a rejected observation into an accepted replay.
                self.last
                    .insert(identity, (measurement.sequence, previous_timestamp_ns));
                return Err(StreamOrderFailure::TimestampMovedBackward {
                    previous_ns: previous_timestamp_ns,
                    received_ns: measurement.captured_at_ns,
                });
            }
        }
        self.last.insert(identity, (measurement.sequence, measurement.captured_at_ns));
        Ok(())
    }
}

fn non_empty(value: String, field: &'static str) -> Result<String, ContractError> {
    let value = value.trim().to_owned();
    if value.is_empty() {
        return Err(ContractError::EmptyIdentifier(field));
    }
    Ok(value)
}
