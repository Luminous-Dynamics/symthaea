// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Firmware-profile-bound evidence for PX4 Offboard-loss parameter values.
//!
//! Raw action integers are preserved, never translated into navigation modes.
//! Missing values are not replaced with assumed defaults. Parameter receive
//! times use a companion clock because MAVLink PARAM_VALUE has no source time.
//! A complete snapshot means captured configuration evidence only: it does not
//! prove source authenticity, configuration correctness, or action execution.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{
    ClockDomainId, EvidenceValidationError, TimestampV1,
};
use thiserror::Error;

pub const PX4_OFFBOARD_LOSS_CONFIGURATION_SCHEMA_V1: u16 = 1;
const OBSERVATION_DOMAIN: &[u8] = b"symthaea.autopilot.px4.offboard-loss-parameter.v1\0";
const PROFILE_DOMAIN: &[u8] = b"symthaea.autopilot.px4.offboard-loss-config-profile.v1\0";
const ASSESSMENT_DOMAIN: &[u8] = b"symthaea.autopilot.px4.offboard-loss-config-assessment.v1\0";

/// Allowlisted parameters relevant to Offboard-loss policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4OffboardLossParameterNameV1 {
    /// Offboard-loss trigger delay in seconds.
    ComOfLossT,
    /// Legacy action value used by some firmware profiles.
    ComOblAct,
    /// Action value whose semantics are firmware-profile-bound.
    ComOblRcAct,
}

impl Px4OffboardLossParameterNameV1 {
    /// Exact PX4 parameter spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ComOfLossT => "COM_OF_LOSS_T",
            Self::ComOblAct => "COM_OBL_ACT",
            Self::ComOblRcAct => "COM_OBL_RC_ACT",
        }
    }
}

/// Typed parameter value preserving the raw timeout bits and action integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "raw", rename_all = "snake_case")]
pub enum Px4OffboardLossParameterValueV1 {
    /// COM_OF_LOSS_T as IEEE-754 binary32 bits (seconds).
    TimeoutSecondsF32Bits(u32),
    /// COM_OBL_ACT as a raw signed integer; no enum mapping is performed.
    LegacyActionI32(i32),
    /// COM_OBL_RC_ACT as a raw signed integer; no enum mapping is performed.
    RcActionI32(i32),
}

impl Px4OffboardLossParameterValueV1 {
    /// Parameter name implied by the wire-value type.
    pub const fn parameter_name(self) -> Px4OffboardLossParameterNameV1 {
        match self {
            Self::TimeoutSecondsF32Bits(_) => Px4OffboardLossParameterNameV1::ComOfLossT,
            Self::LegacyActionI32(_) => Px4OffboardLossParameterNameV1::ComOblAct,
            Self::RcActionI32(_) => Px4OffboardLossParameterNameV1::ComOblRcAct,
        }
    }

    /// Read a timeout as f32 while retaining its committed original bits.
    pub fn timeout_seconds(self) -> Option<f32> {
        match self {
            Self::TimeoutSecondsF32Bits(bits) => Some(f32::from_bits(bits)),
            Self::LegacyActionI32(_) | Self::RcActionI32(_) => None,
        }
    }

    fn validate(self) -> Result<(), Px4OffboardLossConfigurationError> {
        if let Self::TimeoutSecondsF32Bits(bits) = self {
            let seconds = f32::from_bits(bits);
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(Px4OffboardLossConfigurationError::InvalidTimeoutValue);
            }
        }
        Ok(())
    }

    fn feed_digest(self, hasher: &mut blake3::Hasher) {
        match self {
            Self::TimeoutSecondsF32Bits(bits) => {
                hasher.update(&[0]);
                hasher.update(&bits.to_le_bytes());
            }
            Self::LegacyActionI32(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            Self::RcActionI32(value) => {
                hasher.update(&[2]);
                hasher.update(&value.to_le_bytes());
            }
        }
    }
}

/// One parameter response observed on a declared receiver clock.
///
/// The capture sequence is adapter-allocated and unique within a window; it
/// need not be contiguous because parameter responses are not a telemetry stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardLossParameterObservationV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// Parameter transport/decoder identity.
    pub producer_profile_id: String,
    /// Bound vehicle identity, not authenticated by this record.
    pub target_instance_id: String,
    /// Declared firmware/vehicle/parameter-semantics profile.
    pub firmware_profile_id: String,
    /// PX4 boot/session identity supplied by the adapter.
    pub source_boot_session_id: String,
    /// Adapter-allocated unique response sequence.
    pub capture_sequence: u64,
    /// Exact raw parameter value.
    pub value: Px4OffboardLossParameterValueV1,
    /// Local response-receive time in the declared companion clock.
    pub received_at: TimestampV1,
    /// Domain-separated commitment to all preceding fields.
    pub observation_digest_hex: String,
}

impl Px4OffboardLossParameterObservationV1 {
    /// Construct, bind, and validate a parameter response observation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_sequence: u64,
        value: Px4OffboardLossParameterValueV1,
        received_at: TimestampV1,
    ) -> Result<Self, Px4OffboardLossConfigurationError> {
        let mut item = Self {
            schema_version: PX4_OFFBOARD_LOSS_CONFIGURATION_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_sequence,
            value,
            received_at,
            observation_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.observation_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate identifiers, typed value, receive clock, and the content commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardLossConfigurationError> {
        self.validate_without_digest()?;
        if self.observation_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardLossConfigurationError::DigestMismatch("observation"));
        }
        Ok(())
    }

    /// Parameter name implied by the value type.
    pub const fn parameter_name(&self) -> Px4OffboardLossParameterNameV1 {
        self.value.parameter_name()
    }

    /// Content-addressed identity of this exact parameter response.
    pub fn observation_id(&self) -> Result<String, Px4OffboardLossConfigurationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.offboard-loss-parameter.v1:{}",
            self.observation_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardLossConfigurationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        self.value.validate()?;
        self.received_at
            .validate()
            .map_err(Px4OffboardLossConfigurationError::Timestamp)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.firmware_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        hasher.update(&self.capture_sequence.to_le_bytes());
        self.value.feed_digest(&mut hasher);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Firmware-profile-bound completeness, freshness, and capture-skew requirements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardLossConfigurationProfileV1 {
    /// Profile schema version.
    pub schema_version: u16,
    /// Stable policy/profile identifier.
    pub profile_id: String,
    /// Expected vehicle identity.
    pub target_instance_id: String,
    /// Declared firmware/vehicle parameter-semantics profile.
    pub firmware_profile_id: String,
    /// Expected parameter transport/decoder.
    pub producer_profile_id: String,
    /// Expected PX4 boot/session identity.
    pub source_boot_session_id: String,
    /// Clock used to compare parameter receive timestamps.
    pub receiver_clock_domain: ClockDomainId,
    /// Parameters required for this particular firmware profile.
    pub required_parameters: Vec<Px4OffboardLossParameterNameV1>,
    /// Maximum receive-time spread across required parameter responses.
    pub max_capture_skew_ns: u64,
    /// Maximum age permitted for every required parameter response.
    pub max_parameter_age_ns: u64,
    /// Domain-separated profile commitment.
    pub profile_digest_hex: String,
}

impl Px4OffboardLossConfigurationProfileV1 {
    /// Construct a profile with explicit required values and time bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        producer_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        receiver_clock_domain: ClockDomainId,
        mut required_parameters: Vec<Px4OffboardLossParameterNameV1>,
        max_capture_skew_ns: u64,
        max_parameter_age_ns: u64,
    ) -> Result<Self, Px4OffboardLossConfigurationError> {
        required_parameters.sort_unstable();
        let mut profile = Self {
            schema_version: PX4_OFFBOARD_LOSS_CONFIGURATION_SCHEMA_V1,
            profile_id: profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            producer_profile_id: producer_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            receiver_clock_domain,
            required_parameters,
            max_capture_skew_ns,
            max_parameter_age_ns,
            profile_digest_hex: String::new(),
        };
        profile.validate_without_digest()?;
        profile.profile_digest_hex = profile.compute_digest_hex();
        profile.validate()?;
        Ok(profile)
    }

    /// Validate the profile and its commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardLossConfigurationError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardLossConfigurationError::DigestMismatch("profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardLossConfigurationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        self.receiver_clock_domain
            .validate()
            .map_err(Px4OffboardLossConfigurationError::Timestamp)?;
        if !self.required_parameters.contains(&Px4OffboardLossParameterNameV1::ComOfLossT) {
            return Err(Px4OffboardLossConfigurationError::TimeoutParameterNotRequired);
        }
        if self.required_parameters.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Px4OffboardLossConfigurationError::DuplicateRequiredParameter);
        }
        if self.max_capture_skew_ns == 0 {
            return Err(Px4OffboardLossConfigurationError::ZeroCaptureSkew);
        }
        if self.max_parameter_age_ns == 0 {
            return Err(Px4OffboardLossConfigurationError::ZeroParameterAge);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.firmware_profile_id);
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        feed_str(&mut hasher, self.receiver_clock_domain.as_str());
        hasher.update(&(self.required_parameters.len() as u64).to_le_bytes());
        for name in &self.required_parameters {
            feed_str(&mut hasher, name.as_str());
        }
        hasher.update(&self.max_capture_skew_ns.to_le_bytes());
        hasher.update(&self.max_parameter_age_ns.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Conservative result of inspecting required parameter responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Px4OffboardLossConfigurationOutcomeV1 {
    /// Required values were not all observed; no firmware defaults are inferred.
    Incomplete {
        /// Required parameter names absent from this capture.
        missing_parameters: Vec<Px4OffboardLossParameterNameV1>,
    },
    /// At least one required response exceeded the age ceiling.
    Stale {
        /// Age of the oldest required response.
        oldest_parameter_age_ns: u64,
        /// Maximum permitted age.
        max_parameter_age_ns: u64,
    },
    /// Responses are fresh, but their capture times are too far apart.
    CaptureWindowTooWide {
        /// Measured oldest-to-newest receive-time spread.
        capture_skew_ns: u64,
        /// Maximum allowed receive-time spread.
        max_capture_skew_ns: u64,
    },
    /// Complete, fresh required values observed within the capture-skew limit.
    ///
    /// This is not a judgment that the configuration is correct or that it ran.
    CompleteSnapshotObserved {
        /// Exact required raw values and their individual observation identities.
        required_observations: Vec<Px4OffboardLossParameterObservationV1>,
        /// Age of newest required response.
        latest_parameter_age_ns: u64,
        /// Age of oldest required response.
        oldest_parameter_age_ns: u64,
        /// Receive-time spread across required responses.
        capture_skew_ns: u64,
    },
}

/// Content-addressed assessment of one parameter capture window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardLossConfigurationAssessmentV1 {
    /// Assessment schema version.
    pub schema_version: u16,
    /// Exact provenance and capture-window profile.
    pub profile: Px4OffboardLossConfigurationProfileV1,
    /// Receiver-clock time when assessed.
    pub assessed_at: TimestampV1,
    /// Canonically ordered parameter responses.
    pub observations: Vec<Px4OffboardLossParameterObservationV1>,
    /// Domain-separated assessment commitment.
    pub assessment_digest_hex: String,
}

impl Px4OffboardLossConfigurationAssessmentV1 {
    /// Construct, order, bind, and validate a parameter capture window.
    pub fn new(
        profile: Px4OffboardLossConfigurationProfileV1,
        assessed_at: TimestampV1,
        mut observations: Vec<Px4OffboardLossParameterObservationV1>,
    ) -> Result<Self, Px4OffboardLossConfigurationError> {
        observations.sort_by_key(|item| item.parameter_name());
        let mut assessment = Self {
            schema_version: PX4_OFFBOARD_LOSS_CONFIGURATION_SCHEMA_V1,
            profile,
            assessed_at,
            observations,
            assessment_digest_hex: String::new(),
        };
        assessment.validate_without_digest()?;
        assessment.assessment_digest_hex = assessment.compute_digest_hex();
        assessment.validate()?;
        Ok(assessment)
    }

    /// Validate nested records and assessment commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardLossConfigurationError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardLossConfigurationError::DigestMismatch("assessment"));
        }
        Ok(())
    }

    /// Content-addressed identity for this parameter assessment.
    pub fn assessment_id(&self) -> Result<String, Px4OffboardLossConfigurationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.offboard-loss-configuration-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Assess required-value presence, age and capture skew without decoding action integers.
    pub fn outcome(
        &self,
    ) -> Result<Px4OffboardLossConfigurationOutcomeV1, Px4OffboardLossConfigurationError> {
        self.validate()?;
        let mut required = Vec::with_capacity(self.profile.required_parameters.len());
        let mut missing = Vec::new();
        for name in &self.profile.required_parameters {
            match self.observations.iter().find(|item| item.parameter_name() == *name) {
                Some(item) => required.push(item.clone()),
                None => missing.push(*name),
            }
        }
        if !missing.is_empty() {
            return Ok(Px4OffboardLossConfigurationOutcomeV1::Incomplete {
                missing_parameters: missing,
            });
        }

        let mut oldest_age_ns = 0_u64;
        let mut latest_age_ns = u64::MAX;
        let mut first_receive_ns = u64::MAX;
        let mut last_receive_ns = 0_u64;
        for item in &required {
            let age = self
                .assessed_at
                .elapsed_since(&item.received_at)
                .map_err(Px4OffboardLossConfigurationError::Timestamp)?;
            oldest_age_ns = oldest_age_ns.max(age);
            latest_age_ns = latest_age_ns.min(age);
            first_receive_ns = first_receive_ns.min(item.received_at.nanoseconds);
            last_receive_ns = last_receive_ns.max(item.received_at.nanoseconds);
        }
        if oldest_age_ns > self.profile.max_parameter_age_ns {
            return Ok(Px4OffboardLossConfigurationOutcomeV1::Stale {
                oldest_parameter_age_ns: oldest_age_ns,
                max_parameter_age_ns: self.profile.max_parameter_age_ns,
            });
        }
        let capture_skew_ns = last_receive_ns - first_receive_ns;
        if capture_skew_ns > self.profile.max_capture_skew_ns {
            return Ok(Px4OffboardLossConfigurationOutcomeV1::CaptureWindowTooWide {
                capture_skew_ns,
                max_capture_skew_ns: self.profile.max_capture_skew_ns,
            });
        }
        Ok(Px4OffboardLossConfigurationOutcomeV1::CompleteSnapshotObserved {
            required_observations: required,
            latest_parameter_age_ns: latest_age_ns,
            oldest_parameter_age_ns: oldest_age_ns,
            capture_skew_ns,
        })
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardLossConfigurationError> {
        validate_schema(self.schema_version)?;
        self.profile.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4OffboardLossConfigurationError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4OffboardLossConfigurationError::AssessmentClockMismatch);
        }
        let mut names = HashSet::with_capacity(self.observations.len());
        let mut sequences = HashSet::with_capacity(self.observations.len());
        for item in &self.observations {
            item.validate()?;
            if item.producer_profile_id != self.profile.producer_profile_id {
                return Err(Px4OffboardLossConfigurationError::ProducerMismatch);
            }
            if item.target_instance_id != self.profile.target_instance_id {
                return Err(Px4OffboardLossConfigurationError::TargetMismatch);
            }
            if item.firmware_profile_id != self.profile.firmware_profile_id {
                return Err(Px4OffboardLossConfigurationError::FirmwareProfileMismatch);
            }
            if item.source_boot_session_id != self.profile.source_boot_session_id {
                return Err(Px4OffboardLossConfigurationError::BootSessionMismatch);
            }
            if item.received_at.clock_domain != self.profile.receiver_clock_domain {
                return Err(Px4OffboardLossConfigurationError::ReceiverClockMismatch);
            }
            if !names.insert(item.parameter_name()) {
                return Err(Px4OffboardLossConfigurationError::DuplicateParameter);
            }
            if !sequences.insert(item.capture_sequence) {
                return Err(Px4OffboardLossConfigurationError::DuplicateCaptureSequence);
            }
            self.assessed_at.elapsed_since(&item.received_at).map_err(|error| match error {
                EvidenceValidationError::NonMonotonicTimestamp => {
                    Px4OffboardLossConfigurationError::AssessmentBeforeObservation
                }
                other => Px4OffboardLossConfigurationError::Timestamp(other),
            })?;
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        hasher.update(&(self.observations.len() as u64).to_le_bytes());
        for item in &self.observations {
            feed_str(&mut hasher, &item.observation_digest_hex);
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4OffboardLossConfigurationError> {
    if schema_version != PX4_OFFBOARD_LOSS_CONFIGURATION_SCHEMA_V1 {
        return Err(Px4OffboardLossConfigurationError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4OffboardLossConfigurationError> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4OffboardLossConfigurationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Error validating PX4 Offboard-loss parameter evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4OffboardLossConfigurationError {
    /// Unsupported schema version.
    #[error("unsupported Offboard-loss configuration schema version {found}")]
    UnsupportedSchemaVersion { found: u16 },
    /// Identifier is empty, padded, too long, or contains control characters.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Required set omits the timeout parameter.
    #[error("configuration profile must require COM_OF_LOSS_T")]
    TimeoutParameterNotRequired,
    /// Required parameter list contains duplicates.
    #[error("duplicate required Offboard-loss parameter")]
    DuplicateRequiredParameter,
    /// Capture-skew limit must be non-zero.
    #[error("configuration capture-skew limit must be non-zero")]
    ZeroCaptureSkew,
    /// Parameter age limit must be non-zero.
    #[error("configuration parameter-age limit must be non-zero")]
    ZeroParameterAge,
    /// Timeout is NaN, infinite, or negative.
    #[error("COM_OF_LOSS_T must be a finite non-negative Float32")]
    InvalidTimeoutValue,
    /// Source producer did not match profile.
    #[error("parameter producer mismatch")]
    ProducerMismatch,
    /// Target identity did not match profile.
    #[error("parameter target mismatch")]
    TargetMismatch,
    /// Firmware profile did not match.
    #[error("parameter firmware-profile mismatch")]
    FirmwareProfileMismatch,
    /// Boot session did not match.
    #[error("parameter boot-session mismatch")]
    BootSessionMismatch,
    /// Receiver clock did not match.
    #[error("parameter receiver-clock mismatch")]
    ReceiverClockMismatch,
    /// Assessment clock did not match.
    #[error("configuration assessment-clock mismatch")]
    AssessmentClockMismatch,
    /// Duplicate parameter name in one capture window.
    #[error("duplicate parameter in capture window")]
    DuplicateParameter,
    /// Capture sequence was reused.
    #[error("duplicate parameter-response capture sequence")]
    DuplicateCaptureSequence,
    /// Assessment predates a parameter response.
    #[error("assessment precedes parameter observation")]
    AssessmentBeforeObservation,
    /// Timestamp or clock validation failed.
    #[error("invalid configuration timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Committed data was mutated.
    #[error("configuration {0} digest mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> ClockDomainId {
        ClockDomainId::new("companion.monotonic").unwrap()
    }

    fn ts(ns: u64) -> TimestampV1 {
        TimestampV1::new(clock(), ns)
    }

    fn observation(
        sequence: u64,
        value: Px4OffboardLossParameterValueV1,
        received_ns: u64,
    ) -> Px4OffboardLossParameterObservationV1 {
        Px4OffboardLossParameterObservationV1::new(
            "px4.param.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-profile.fixture.v1",
            "px4.boot.fixture.session-a",
            sequence,
            value,
            ts(received_ns),
        )
        .unwrap()
    }

    fn timeout(sequence: u64, at: u64) -> Px4OffboardLossParameterObservationV1 {
        observation(
            sequence,
            Px4OffboardLossParameterValueV1::TimeoutSecondsF32Bits(1.0_f32.to_bits()),
            at,
        )
    }

    fn action(sequence: u64, raw: i32, at: u64) -> Px4OffboardLossParameterObservationV1 {
        observation(sequence, Px4OffboardLossParameterValueV1::RcActionI32(raw), at)
    }

    fn profile(max_skew_ns: u64) -> Px4OffboardLossConfigurationProfileV1 {
        Px4OffboardLossConfigurationProfileV1::new(
            "px4.offboard-loss.config.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-profile.fixture.v1",
            "px4.param.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            clock(),
            vec![
                Px4OffboardLossParameterNameV1::ComOfLossT,
                Px4OffboardLossParameterNameV1::ComOblRcAct,
            ],
            max_skew_ns,
            2_000_000_000,
        )
        .unwrap()
    }

    fn assessment(
        observations: Vec<Px4OffboardLossParameterObservationV1>,
        assessed_ns: u64,
    ) -> Px4OffboardLossConfigurationAssessmentV1 {
        Px4OffboardLossConfigurationAssessmentV1::new(
            profile(200_000_000),
            ts(assessed_ns),
            observations,
        )
        .unwrap()
    }

    #[test]
    fn complete_snapshot_preserves_raw_action_without_decoding() {
        let value = assessment(
            vec![action(12, 6, 1_100_000_000), timeout(11, 1_000_000_000)],
            1_200_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4OffboardLossConfigurationOutcomeV1::CompleteSnapshotObserved {
                required_observations,
                capture_skew_ns: 100_000_000,
                ..
            } if required_observations.iter().any(|entry|
                entry.value == Px4OffboardLossParameterValueV1::RcActionI32(6))
        ));
        assert!(value.assessment_id().unwrap().starts_with(
            "symthaea.autopilot.px4.offboard-loss-configuration-assessment.v1:"
        ));
    }

    #[test]
    fn missing_action_parameter_is_incomplete_not_defaulted() {
        let value = assessment(vec![timeout(11, 1_000_000_000)], 1_100_000_000);
        assert_eq!(
            value.outcome().unwrap(),
            Px4OffboardLossConfigurationOutcomeV1::Incomplete {
                missing_parameters: vec![Px4OffboardLossParameterNameV1::ComOblRcAct],
            }
        );
    }

    #[test]
    fn stale_required_value_makes_the_whole_window_stale() {
        let value = assessment(
            vec![timeout(11, 1_000_000_000), action(12, 6, 1_100_000_000)],
            4_000_000_001,
        );
        assert_eq!(
            value.outcome().unwrap(),
            Px4OffboardLossConfigurationOutcomeV1::Stale {
                oldest_parameter_age_ns: 3_000_000_001,
                max_parameter_age_ns: 2_000_000_000,
            }
        );
    }

    #[test]
    fn overly_wide_capture_window_is_not_complete() {
        let value = Px4OffboardLossConfigurationAssessmentV1::new(
            profile(200),
            ts(1_300),
            vec![timeout(11, 1_000), action(12, 6, 1_201)],
        ).unwrap();
        assert_eq!(
            value.outcome().unwrap(),
            Px4OffboardLossConfigurationOutcomeV1::CaptureWindowTooWide {
                capture_skew_ns: 201,
                max_capture_skew_ns: 200,
            }
        );
    }

    #[test]
    fn invalid_timeout_is_rejected_and_raw_action_integer_is_preserved() {
        assert_eq!(
            Px4OffboardLossParameterObservationV1::new(
                "px4.param.decoder.fixture.v1",
                "vehicle.slot-a",
                "px4.firmware-profile.fixture.v1",
                "px4.boot.fixture.session-a",
                1,
                Px4OffboardLossParameterValueV1::TimeoutSecondsF32Bits(f32::NAN.to_bits()),
                ts(1_000),
            ).unwrap_err(),
            Px4OffboardLossConfigurationError::InvalidTimeoutValue
        );
        assert_eq!(
            action(12, i32::MAX, 1_100).value,
            Px4OffboardLossParameterValueV1::RcActionI32(i32::MAX)
        );
    }

    #[test]
    fn boot_session_mismatch_fails_closed() {
        let wrong_session = Px4OffboardLossParameterObservationV1::new(
            "px4.param.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-profile.fixture.v1",
            "px4.boot.fixture.session-b",
            12,
            Px4OffboardLossParameterValueV1::RcActionI32(6),
            ts(1_100_000_000),
        ).unwrap();
        assert_eq!(
            Px4OffboardLossConfigurationAssessmentV1::new(
                profile(200_000_000),
                ts(1_200_000_000),
                vec![timeout(11, 1_000_000_000), wrong_session],
            ).unwrap_err(),
            Px4OffboardLossConfigurationError::BootSessionMismatch
        );
    }

    #[test]
    fn canonical_order_produces_same_assessment_identity() {
        let first = assessment(
            vec![timeout(11, 1_000_000_000), action(12, 6, 1_100_000_000)],
            1_200_000_000,
        );
        let second = assessment(
            vec![action(12, 6, 1_100_000_000), timeout(11, 1_000_000_000)],
            1_200_000_000,
        );
        assert_eq!(first.assessment_id().unwrap(), second.assessment_id().unwrap());
    }

    #[test]
    fn serde_mutation_breaks_content_commitments() {
        let value = assessment(
            vec![timeout(11, 1_000_000_000), action(12, 6, 1_100_000_000)],
            1_200_000_000,
        );
        let encoded = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4OffboardLossConfigurationAssessmentV1 =
            serde_json::from_slice(&encoded).unwrap();
        restored.validate().unwrap();
        restored.observations[0].value = Px4OffboardLossParameterValueV1::RcActionI32(0);
        assert!(restored.validate().is_err());
    }
}
