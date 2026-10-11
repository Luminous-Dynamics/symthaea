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

mod certificate;
pub use certificate::{
    CALIBRATION_CERTIFICATE_SCHEMA, CalibrationCertificate, CalibrationCertificateError,
};
mod evidence;
pub use evidence::{
    ArtifactVerificationError, ImmutableArtifactReader, VerifiedArtifactBytes,
    VerifiedArtifactDigest, MAX_IMMUTABLE_SOURCE_VERSION_ID_BYTES, sha256_hex,
};
mod review;
pub use review::{
    CALIBRATION_REVIEW_SCHEMA, REVIEW_TRUST_STORE_SCHEMA, CalibrationReviewPayload,
    ReviewAuthorizationPolicy, ReviewTrustKey, ReviewTrustStore, ReviewTrustStoreGuard,
    ReviewVerificationError,
    SignedCalibrationReviewReceipt, SignedReviewTrustStoreSnapshot,
    VerifiedCalibrationReviewReceipt, VerifiedReviewerAuthorization, VerifiedReviewTrustStore,
};

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

/// Identifier for a specific monotonic clock domain/epoch. Use a new identifier
/// when the clock origin is reset (for example, after reboot). A timestamp from
/// one domain must never be compared to "now" from another domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClockDomainId(String);

impl ClockDomainId {
    pub fn new(value: impl Into<String>) -> Result<Self, ContractError> {
        Ok(Self(non_empty(value.into(), "clock_domain_id")?))
    }

    pub fn as_str(&self) -> &str {
        &self.0
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
    OxygenSaturationPercent,
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
            Self::Percent => Quantity::Dimensionless,
            Self::OxygenSaturationPercent => Quantity::OxygenSaturation,
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

/// Raw acquisition metadata returned only after an immutable version-bound reader
/// has been hashed against the referenced digest. A digest-only read from a mutable
/// path cannot be promoted into resolved evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRawData {
    reference: ArtifactReference,
    byte_length: u64,
    source_version_id: String,
    media_type: String,
}

impl ResolvedRawData {
    /// Build raw-data metadata only from a digest verified against an immutable read.
    ///
    /// Integrity-only results from verify_reader carry no source version and cannot
    /// be promoted into resolved raw evidence.
    pub fn from_verified_digest(
        verified: VerifiedArtifactDigest,
        media_type: impl Into<String>,
    ) -> Result<Self, ContractError> {
        if verified.byte_length() == 0 {
            return Err(ContractError::EmptyRawDataArtifact);
        }
        let media_type = non_empty(media_type.into(), "raw_data_media_type")?;
        let source_version_id = verified
            .immutable_source_version_id()
            .ok_or(ContractError::ImmutableSourceVersionRequired)?
            .to_owned();
        Ok(Self {
            reference: verified.reference().clone(),
            byte_length: verified.byte_length(),
            source_version_id,
            media_type,
        })
    }

    pub fn artifact(&self) -> &ArtifactReference {
        &self.reference
    }

    pub fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Version token that must be used if the resolver later reopens this artifact.
    pub fn source_version_id(&self) -> &str {
        &self.source_version_id
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
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
    EmptyRawDataArtifact,
    ImmutableSourceVersionRequired,
    InvalidValidityInterval,
    CalibrationReviewMustBeDistinct,
    CalibrationReviewAuthorizationMissing,
    CalibrationReviewAuthorizationMismatch,
    VerifiedArtifactReferenceMismatch(&'static str),
    VerifiedReviewReceiptBindingMismatch,
    InvalidCalibrationCertificate(String),
    CalibrationCertificateClaimMismatch(&'static str),
    InvalidReferenceStandardCertificate(String),
    ReferenceStandardClaimMismatch(&'static str),
    EmptyEvidenceArtifact(&'static str),
    InvalidCalibrationRange,
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
            Self::EmptyRawDataArtifact => {
                write!(f, "raw-data evidence artifact must not be empty")
            }
            Self::ImmutableSourceVersionRequired => write!(
                f,
                "raw evidence requires an immutable source-version-bound read"
            ),
            Self::InvalidValidityInterval => {
                write!(f, "validity interval must satisfy start < end")
            }
            Self::CalibrationReviewMustBeDistinct => {
                write!(f, "calibration evidence and review receipt must be distinct artifacts")
            }
            Self::CalibrationReviewAuthorizationMissing => write!(
                f,
                "calibration evidence requires reviewer authorization proof"
            ),
            Self::CalibrationReviewAuthorizationMismatch => write!(
                f,
                "review authorization provider or organization mismatch"
            ),
            Self::VerifiedArtifactReferenceMismatch(artifact) => {
                write!(f, "verified bytes must match the resolved {artifact} reference")
            }
            Self::VerifiedReviewReceiptBindingMismatch => write!(
                f,
                "verified review receipt must bind the resolved calibration record and certificate"
            ),
            Self::InvalidCalibrationCertificate(reason) => {
                write!(f, "calibration certificate is invalid: {reason}")
            }
            Self::CalibrationCertificateClaimMismatch(field) => {
                write!(f, "resolved calibration claim does not match certificate field {field}")
            }
            Self::InvalidReferenceStandardCertificate(reason) => {
                write!(f, "reference-standard certificate is invalid: {reason}")
            }
            Self::ReferenceStandardClaimMismatch(field) => {
                write!(f, "reference-standard certificate claim does not match {field}")
            }
            Self::EmptyEvidenceArtifact(artifact) => {
                write!(f, "{artifact} evidence must not be empty")
            }
            Self::InvalidCalibrationRange => {
                write!(f, "calibration minimum must not exceed its maximum")
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
    clock_domain: ClockDomainId,
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
    /// Monotonic acquisition timestamp in nanoseconds, from the named clock domain.
    pub captured_at_ns: u64,
    pub clock_domain: ClockDomainId,
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
            clock_domain: input.clock_domain,
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

    pub fn clock_domain(&self) -> &ClockDomainId {
        &self.clock_domain
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
        now_clock_domain: &ClockDomainId,
        policy: &MeasurementPolicy,
        resolver: &dyn InstrumentEvidenceResolver,
        stream_guard: &mut MeasurementStreamGuard,
    ) -> Result<MeasurementAssessment, AssessmentFailure> {
        if &self.clock_domain != now_clock_domain {
            return Err(AssessmentFailure::ClockDomainMismatch);
        }
        if policy.uncertainty_unit != self.unit {
            return Err(AssessmentFailure::PolicyUnitMismatch {
                policy_unit: policy.uncertainty_unit,
                measurement_unit: self.unit,
            });
        }
        stream_guard
            .observe_at(self, now_ns)
            .map_err(|failure| match failure {
                StreamOrderFailure::FutureTimestamp { .. } => AssessmentFailure::ClockInFuture,
                other => AssessmentFailure::StreamOrder(other),
            })?;
        let age_ns = now_ns - self.captured_at_ns;
        if age_ns > policy.max_age_ns {
            return Err(AssessmentFailure::Stale { age_ns, max_age_ns: policy.max_age_ns });
        }
        if !self.quality_flags.is_empty() {
            return Err(AssessmentFailure::QualityFlagsPresent(self.quality_flags.clone()));
        }
        let raw_reference = self
            .raw_data
            .as_ref()
            .ok_or(AssessmentFailure::MissingRawDataReference)?;
        let reference = self
            .calibration
            .as_ref()
            .ok_or(AssessmentFailure::MissingCalibrationReference)?;

        let resolved_raw = resolver
            .resolve_raw_data(raw_reference)
            .map_err(AssessmentFailure::RawDataEvidenceUnresolved)?;
        if resolved_raw.artifact() != raw_reference.artifact() {
            return Err(AssessmentFailure::RawDataReferenceMismatch);
        }

        let verified_calibration = resolver
            .resolve_calibration(reference)
            .map_err(AssessmentFailure::CalibrationEvidenceUnresolved)?;
        let resolved = verified_calibration.calibration();

        if resolved.record_id != reference.record_id
            || resolved.evidence != reference.evidence
        {
            return Err(AssessmentFailure::CalibrationReferenceMismatch);
        }
        if resolved.instrument != self.identity {
            return Err(AssessmentFailure::CalibrationInstrumentMismatch);
        }
        if resolved.quantity != self.quantity || resolved.unit != self.unit {
            return Err(AssessmentFailure::CalibrationUnitMismatch);
        }
        if !(resolved.valid_from_ns <= self.captured_at_ns
            && self.captured_at_ns < resolved.valid_until_ns)
        {
            return Err(AssessmentFailure::CalibrationNotValidAtCapture);
        }
        if self.value < resolved.range_min_value || self.value > resolved.range_max_value {
            return Err(AssessmentFailure::CalibrationRangeExceeded {
                value: self.value,
                minimum: resolved.range_min_value,
                maximum: resolved.range_max_value,
            });
        }
        if self.standard_uncertainty > policy.max_measurement_standard_uncertainty {
            return Err(AssessmentFailure::MeasurementUncertaintyExceeded);
        }
        if resolved.standard_uncertainty > policy.max_calibration_standard_uncertainty {
            return Err(AssessmentFailure::CalibrationUncertaintyExceeded);
        }
        let certificate_claims = verified_calibration.certificate_claims();
        let reviewer_authorization = verified_calibration
            .review_receipt()
            .reviewer_authorization()
            .cloned()
            .ok_or(AssessmentFailure::CalibrationReviewAuthorizationMissing)?;
        let calibration_provider_id =
            certificate_claims.calibration_provider_id().to_owned();
        let calibration_procedure_id =
            certificate_claims.calibration_procedure_id().to_owned();
        let calibration_reference_standard = verified_calibration
            .reference_standard()
            .reference()
            .clone();

        Ok(MeasurementAssessment {
            instrument: self.identity.clone(),
            sequence: self.sequence,
            captured_at_ns: self.captured_at_ns,
            clock_domain: self.clock_domain.clone(),
            quantity: self.quantity,
            unit: self.unit,
            value: self.value,
            age_ns,
            measurement_standard_uncertainty: self.standard_uncertainty,
            calibration_standard_uncertainty: resolved.standard_uncertainty,
            calibration_record_id: resolved.record_id.clone(),
            calibration_evidence: resolved.evidence.clone(),
            calibration_review_receipt: resolved.review_receipt.clone(),
            calibration_provider_id,
            calibration_procedure_id,
            calibration_reference_standard,
            calibration_review_authorization: reviewer_authorization,
            calibration_range_min_value: resolved.range_min_value,
            calibration_range_max_value: resolved.range_max_value,
            raw_data: resolved_raw,
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
    instrument: InstrumentIdentity,
    evidence: ArtifactReference,
    review_receipt: ArtifactReference,
    quantity: Quantity,
    unit: Unit,
    range_min_value: f64,
    range_max_value: f64,
    valid_from_ns: u64,
    valid_until_ns: u64,
    standard_uncertainty: f64,
}

/// A calibration result bound to an exact instrument/channel, supported quantity/unit,
/// inclusive value range, validity interval, and standard uncertainty. Range bounds are
/// expressed in `unit`; the end of the validity interval is exclusive.
impl ResolvedCalibration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        record_id: impl Into<String>,
        instrument: InstrumentIdentity,
        evidence: ArtifactReference,
        review_receipt: ArtifactReference,
        quantity: Quantity,
        unit: Unit,
        range_min_value: f64,
        range_max_value: f64,
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
        if !range_min_value.is_finite() {
            return Err(ContractError::NonFiniteValue("calibration_range_min_value"));
        }
        if !range_max_value.is_finite() {
            return Err(ContractError::NonFiniteValue("calibration_range_max_value"));
        }
        if range_min_value > range_max_value {
            return Err(ContractError::InvalidCalibrationRange);
        }
        if !standard_uncertainty.is_finite() {
            return Err(ContractError::NonFiniteValue("calibration_standard_uncertainty"));
        }
        if standard_uncertainty < 0.0 {
            return Err(ContractError::NegativeUncertainty("calibration_standard_uncertainty"));
        }
        Ok(Self {
            record_id: non_empty(record_id.into(), "calibration_record_id")?,
            instrument,
            evidence,
            review_receipt,
            quantity,
            unit,
            range_min_value,
            range_max_value,
            valid_from_ns,
            valid_until_ns,
            standard_uncertainty,
        })
    }

    pub fn evidence(&self) -> &ArtifactReference {
        &self.evidence
    }

    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    pub fn instrument(&self) -> &InstrumentIdentity {
        &self.instrument
    }

    pub fn quantity(&self) -> Quantity {
        self.quantity
    }

    pub fn unit(&self) -> Unit {
        self.unit
    }

    pub fn range(&self) -> (f64, f64) {
        (self.range_min_value, self.range_max_value)
    }

    pub fn validity_window_ns(&self) -> (u64, u64) {
        (self.valid_from_ns, self.valid_until_ns)
    }

    pub fn standard_uncertainty(&self) -> f64 {
        self.standard_uncertainty
    }

    pub fn review_receipt(&self) -> &ArtifactReference {
        &self.review_receipt
    }
}

/// Calibration metadata paired with byte-verified, claim-bound evidence.
///
/// Construction parses the canonical certificate and checks its record,
/// instrument/channel, quantity/unit, range, validity window, and uncertainty
/// against the resolved record. It requires verified bytes for the certificate's
/// referenced standard and a signed review receipt bound to the record and exact
/// certificate artifact. It does not authenticate the trust-store distribution,
/// authorize the provider, or validate a complete traceability chain.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedCalibrationEvidence {
    calibration: ResolvedCalibration,
    certificate: VerifiedArtifactBytes,
    certificate_claims: CalibrationCertificate,
    reference_standard: VerifiedArtifactBytes,
    reference_standard_claims: CalibrationCertificate,
    review_receipt: VerifiedCalibrationReviewReceipt,
}

impl VerifiedCalibrationEvidence {
    pub fn new(
        calibration: ResolvedCalibration,
        certificate: VerifiedArtifactBytes,
        reference_standard: VerifiedArtifactBytes,
        review_receipt: VerifiedCalibrationReviewReceipt,
    ) -> Result<Self, ContractError> {
        if certificate.byte_length() == 0 {
            return Err(ContractError::EmptyEvidenceArtifact(
                "calibration_certificate",
            ));
        }
        if review_receipt.artifact().byte_length() == 0 {
            return Err(ContractError::EmptyEvidenceArtifact(
                "calibration_review_receipt",
            ));
        }
        if certificate.reference() != &calibration.evidence {
            return Err(ContractError::VerifiedArtifactReferenceMismatch(
                "calibration_certificate",
            ));
        }
        let parsed_certificate = CalibrationCertificate::from_canonical_bytes(
            certificate.bytes(),
        )
        .map_err(|error| ContractError::InvalidCalibrationCertificate(error.to_string()))?;
        let (range_min_value, range_max_value) = calibration.range();
        let (valid_from_ns, valid_until_ns) = calibration.validity_window_ns();
        if let Some(field) = parsed_certificate.first_metadata_mismatch(
            calibration.record_id(),
            calibration.instrument(),
            calibration.quantity(),
            calibration.unit(),
            range_min_value,
            range_max_value,
            valid_from_ns,
            valid_until_ns,
            calibration.standard_uncertainty(),
        ) {
            return Err(ContractError::CalibrationCertificateClaimMismatch(field));
        }
        let reviewer_authorization = review_receipt
            .reviewer_authorization()
            .ok_or(ContractError::CalibrationReviewAuthorizationMissing)?;
        if reviewer_authorization.provider_organization_id()
            != parsed_certificate.calibration_provider_id()
            || reviewer_authorization.reviewer_organization_id()
                == parsed_certificate.calibration_provider_id()
        {
            return Err(ContractError::CalibrationReviewAuthorizationMismatch);
        }
        if reference_standard.byte_length() == 0 {
            return Err(ContractError::EmptyEvidenceArtifact("reference_standard"));
        }
        if reference_standard.reference() != parsed_certificate.reference_standard() {
            return Err(ContractError::VerifiedArtifactReferenceMismatch(
                "reference_standard",
            ));
        }
        let parsed_reference_standard = CalibrationCertificate::from_canonical_bytes(
            reference_standard.bytes(),
        )
        .map_err(|error| {
            ContractError::InvalidReferenceStandardCertificate(error.to_string())
        })?;
        if parsed_reference_standard.record_id()
            != parsed_certificate.reference_standard().artifact_id()
        {
            return Err(ContractError::ReferenceStandardClaimMismatch("record_id"));
        }
        if parsed_reference_standard.quantity() != parsed_certificate.quantity() {
            return Err(ContractError::ReferenceStandardClaimMismatch("quantity"));
        }
        if parsed_reference_standard.unit() != parsed_certificate.unit() {
            return Err(ContractError::ReferenceStandardClaimMismatch("unit"));
        }
        let (standard_min, standard_max) = parsed_reference_standard.range();
        let (calibration_min, calibration_max) = parsed_certificate.range();
        if standard_min > calibration_min || standard_max < calibration_max {
            return Err(ContractError::ReferenceStandardClaimMismatch(
                "calibration_range_not_covered",
            ));
        }
        let (standard_valid_from, standard_valid_until) =
            parsed_reference_standard.validity_window_ns();
        let (calibration_valid_from, calibration_valid_until) =
            parsed_certificate.validity_window_ns();
        if standard_valid_from > calibration_valid_from
            || calibration_valid_until > standard_valid_until
        {
            return Err(ContractError::ReferenceStandardClaimMismatch(
                "validity_interval_not_covered",
            ));
        }
        if review_receipt.artifact().reference() != &calibration.review_receipt {
            return Err(ContractError::VerifiedArtifactReferenceMismatch(
                "calibration_review_receipt",
            ));
        }
        if review_receipt.payload().record_id() != calibration.record_id
            || review_receipt.payload().certificate_artifact_id()
                != calibration.evidence.artifact_id()
            || review_receipt.payload().certificate_sha256_hex()
                != calibration.evidence.sha256_hex()
        {
            return Err(ContractError::VerifiedReviewReceiptBindingMismatch);
        }
        Ok(Self {
            calibration,
            certificate,
            certificate_claims: parsed_certificate,
            reference_standard,
            reference_standard_claims: parsed_reference_standard,
            review_receipt,
        })
    }

    pub fn calibration(&self) -> &ResolvedCalibration {
        &self.calibration
    }

    pub fn certificate(&self) -> &VerifiedArtifactBytes {
        &self.certificate
    }

    pub fn certificate_claims(&self) -> &CalibrationCertificate {
        &self.certificate_claims
    }

    pub fn reference_standard(&self) -> &VerifiedArtifactBytes {
        &self.reference_standard
    }

    pub fn reference_standard_claims(&self) -> &CalibrationCertificate {
        &self.reference_standard_claims
    }

    pub fn review_receipt(&self) -> &VerifiedCalibrationReviewReceipt {
        &self.review_receipt
    }
}

/// Resolve a calibration record and its evidence from an independently governed
/// source. Implementations must hash the exact certificate, reference-standard,
/// and signed review-receipt bytes and return them through
/// `VerifiedCalibrationEvidence`. The wrapper parses and binds the certificate's
/// declared claims, but the resolver/deployment must authenticate trust-store
/// distribution, provider authority, reference-standard provenance, and the full
/// traceability chain. The type cannot prove resolver independence by itself.
pub trait CalibrationEvidenceResolver {
    fn resolve_calibration(
        &self,
        reference: &CalibrationReference,
    ) -> Result<VerifiedCalibrationEvidence, String>;
}

/// Resolve raw acquisition bytes from an immutable reference and independently
/// verify that their SHA-256 digest matches the reference. Return metadata only
/// after the bytes have been read/verified; a database row lookup is insufficient.
pub trait RawDataEvidenceResolver {
    fn resolve_raw_data(
        &self,
        reference: &RawDataReference,
    ) -> Result<ResolvedRawData, String>;
}

/// The quantitative-use gate requires both trust-boundary resolvers. The same
/// adapter may implement both traits, but deployments should document how raw
/// byte integrity and calibration authority are independently established.
pub trait InstrumentEvidenceResolver: CalibrationEvidenceResolver + RawDataEvidenceResolver {}

impl<T> InstrumentEvidenceResolver for T where
    T: CalibrationEvidenceResolver + RawDataEvidenceResolver
{
}

/// Conservative thresholds for a particular consumer and intended use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementPolicy {
    max_age_ns: u64,
    uncertainty_unit: Unit,
    max_measurement_standard_uncertainty: f64,
    max_calibration_standard_uncertainty: f64,
}

impl MeasurementPolicy {
    pub fn new(
        max_age_ns: u64,
        uncertainty_unit: Unit,
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
            uncertainty_unit,
            max_measurement_standard_uncertainty,
            max_calibration_standard_uncertainty,
        })
    }

    pub fn uncertainty_unit(&self) -> Unit {
        self.uncertainty_unit
    }
}

/// Immutable-by-interface result of a successful quantitative-use assessment.
/// Fields are private so downstream code cannot create a forged assessment with
/// a struct literal. This is still not a signed receipt or clinical authorization.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementAssessment {
    instrument: InstrumentIdentity,
    sequence: u64,
    captured_at_ns: u64,
    clock_domain: ClockDomainId,
    quantity: Quantity,
    unit: Unit,
    value: f64,
    age_ns: u64,
    measurement_standard_uncertainty: f64,
    calibration_standard_uncertainty: f64,
    calibration_record_id: String,
    calibration_evidence: ArtifactReference,
    calibration_review_receipt: ArtifactReference,
    calibration_provider_id: String,
    calibration_procedure_id: String,
    calibration_reference_standard: ArtifactReference,
    calibration_review_authorization: VerifiedReviewerAuthorization,
    calibration_range_min_value: f64,
    calibration_range_max_value: f64,
    raw_data: ResolvedRawData,
    processing_chain_version: String,
}

impl MeasurementAssessment {
    pub fn instrument(&self) -> &InstrumentIdentity { &self.instrument }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn captured_at_ns(&self) -> u64 { self.captured_at_ns }
    pub fn clock_domain(&self) -> &ClockDomainId { &self.clock_domain }
    pub fn quantity(&self) -> Quantity { self.quantity }
    pub fn unit(&self) -> Unit { self.unit }
    pub fn value(&self) -> f64 { self.value }
    pub fn age_ns(&self) -> u64 { self.age_ns }
    pub fn measurement_standard_uncertainty(&self) -> f64 {
        self.measurement_standard_uncertainty
    }
    pub fn calibration_standard_uncertainty(&self) -> f64 {
        self.calibration_standard_uncertainty
    }
    pub fn calibration_record_id(&self) -> &str { &self.calibration_record_id }
    pub fn calibration_evidence(&self) -> &ArtifactReference { &self.calibration_evidence }
    pub fn calibration_review_receipt(&self) -> &ArtifactReference {
        &self.calibration_review_receipt
    }
    pub fn calibration_provider_id(&self) -> &str {
        &self.calibration_provider_id
    }
    pub fn calibration_procedure_id(&self) -> &str {
        &self.calibration_procedure_id
    }
    pub fn calibration_reference_standard(&self) -> &ArtifactReference {
        &self.calibration_reference_standard
    }
    pub fn calibration_review_authorization(&self) -> &VerifiedReviewerAuthorization {
        &self.calibration_review_authorization
    }
    pub fn calibration_range(&self) -> (f64, f64) {
        (self.calibration_range_min_value, self.calibration_range_max_value)
    }
    pub fn raw_data(&self) -> &ResolvedRawData { &self.raw_data }
    pub fn processing_chain_version(&self) -> &str { &self.processing_chain_version }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssessmentFailure {
    StreamOrder(StreamOrderFailure),
    ClockInFuture,
    ClockDomainMismatch,
    Stale { age_ns: u64, max_age_ns: u64 },
    QualityFlagsPresent(Vec<QualityFlag>),
    MissingRawDataReference,
    MissingCalibrationReference,
    RawDataEvidenceUnresolved(String),
    RawDataReferenceMismatch,
    CalibrationEvidenceUnresolved(String),
    CalibrationReviewAuthorizationMissing,
    CalibrationReferenceMismatch,
    CalibrationInstrumentMismatch,
    CalibrationUnitMismatch,
    CalibrationNotValidAtCapture,
    CalibrationRangeExceeded { value: f64, minimum: f64, maximum: f64 },
    PolicyUnitMismatch { policy_unit: Unit, measurement_unit: Unit },
    MeasurementUncertaintyExceeded,
    CalibrationUncertaintyExceeded,
}

impl fmt::Display for AssessmentFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AssessmentFailure {}

/// Per-instrument/channel guard against replayed, duplicate, future-dated, or
/// time-reversed observations. Rejected new sequences are consumed while the
/// last accepted timestamp remains monotonic, preventing retry-based replay and
/// preventing a future-dated sample from poisoning the timestamp high-water mark.
#[derive(Debug, Default)]
pub struct MeasurementStreamGuard {
    last: HashMap<InstrumentIdentity, StreamPosition>,
}

#[derive(Debug, Clone)]
struct StreamPosition {
    sequence: u64,
    clock_domain: ClockDomainId,
    last_timestamp_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOrderFailure {
    SequenceNotIncreasing { previous: u64, received: u64 },
    ClockDomainChanged {
        previous: ClockDomainId,
        received: ClockDomainId,
    },
    TimestampMovedBackward { previous_ns: u64, received_ns: u64 },
    FutureTimestamp { captured_at_ns: u64, now_ns: u64 },
}

impl MeasurementStreamGuard {
    /// Check sequence/time ordering only. This method does not assess freshness,
    /// resolve evidence, or qualify a measurement for quantitative use. Callers must
    /// use `MeasurementEnvelope::assess_for_quantitative_use` for the complete gate.
    pub fn observe(&mut self, measurement: &MeasurementEnvelope) -> Result<(), StreamOrderFailure> {
        self.observe_inner(measurement, None)
    }

    /// Validate ordering and reject future timestamps using the same clock basis
    /// represented by `now_ns`. A future timestamp consumes the sequence but
    /// retains the last known non-future timestamp, so it cannot freeze the channel.
    fn observe_at(
        &mut self,
        measurement: &MeasurementEnvelope,
        now_ns: u64,
    ) -> Result<(), StreamOrderFailure> {
        self.observe_inner(measurement, Some(now_ns))
    }

    fn observe_inner(
        &mut self,
        measurement: &MeasurementEnvelope,
        now_ns: Option<u64>,
    ) -> Result<(), StreamOrderFailure> {
        let identity = measurement.identity.clone();
        let previous = self.last.get(&identity).cloned();

        if let Some(position) = previous.as_ref() {
            if position.clock_domain != measurement.clock_domain {
                return Err(StreamOrderFailure::ClockDomainChanged {
                    previous: position.clock_domain.clone(),
                    received: measurement.clock_domain.clone(),
                });
            }
            if measurement.sequence <= position.sequence {
                return Err(StreamOrderFailure::SequenceNotIncreasing {
                    previous: position.sequence,
                    received: measurement.sequence,
                });
            }

            if let Some(previous_timestamp_ns) = position.last_timestamp_ns {
                if measurement.captured_at_ns < previous_timestamp_ns {
                    self.last.insert(
                        identity,
                        StreamPosition {
                            sequence: measurement.sequence,
                            clock_domain: measurement.clock_domain.clone(),
                            last_timestamp_ns: Some(previous_timestamp_ns),
                        },
                    );
                    return Err(StreamOrderFailure::TimestampMovedBackward {
                        previous_ns: previous_timestamp_ns,
                        received_ns: measurement.captured_at_ns,
                    });
                }
            }
        }

        if let Some(now_ns) = now_ns {
            if measurement.captured_at_ns > now_ns {
                self.last.insert(
                    identity,
                    StreamPosition {
                        sequence: measurement.sequence,
                        clock_domain: measurement.clock_domain.clone(),
                        last_timestamp_ns: previous
                            .as_ref()
                            .and_then(|p| p.last_timestamp_ns),
                    },
                );
                return Err(StreamOrderFailure::FutureTimestamp {
                    captured_at_ns: measurement.captured_at_ns,
                    now_ns,
                });
            }
        }

        self.last.insert(
            identity,
            StreamPosition {
                sequence: measurement.sequence,
                clock_domain: measurement.clock_domain.clone(),
                last_timestamp_ns: Some(measurement.captured_at_ns),
            },
        );
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
