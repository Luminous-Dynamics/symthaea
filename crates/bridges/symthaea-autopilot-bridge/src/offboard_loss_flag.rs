// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PX4 FailsafeFlags observations for the Offboard control-signal-loss input.
//!
//! This module preserves the specific offboard_control_signal_lost input flag
//! separately from VehicleStatus.failsafe. The former is an input to PX4's
//! failsafe state machine; it is not, on its own, proof of the resulting mode,
//! configured response, message authenticity, or physical safety.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, EvidenceValidationError, TimestampV1};
use thiserror::Error;

/// Schema version for PX4 Offboard-loss-flag evidence.
pub const PX4_OFFBOARD_LOSS_FLAG_SCHEMA_V1: u16 = 1;
const OBSERVATION_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.offboard-loss-flag.v1\0";
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.offboard-loss-profile.v1\0";
const ASSESSMENT_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.offboard-loss-assessment.v1\0";

/// One captured, decoded FailsafeFlags publication.
///
/// Source time is PX4 boot-relative microseconds. Receive time is a different,
/// receiver-local clock. Content addressing detects record mutation but does not
/// authenticate the telemetry producer or prove the payload came from the vehicle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardLossFlagObservationV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// Decoder/telemetry producer profile identity.
    pub producer_profile_id: String,
    /// Upstream-bound vehicle/autopilot instance identity; not authenticated here.
    pub target_instance_id: String,
    /// Declared controller firmware, vehicle class, and message-schema profile.
    pub controller_profile_id: String,
    /// PX4 boot/session identity for comparing source timestamps.
    pub source_boot_session_id: String,
    /// Capture sequence allocated to each accepted source publication.
    pub capture_sequence: u64,
    /// Raw PX4 FailsafeFlags timestamp, microseconds since system start.
    pub source_timestamp_us: u64,
    /// Raw mode_req_offboard_signal bitfield, preserved without reinterpretation.
    pub mode_req_offboard_signal_raw: u32,
    /// PX4 FailsafeFlags offboard_control_signal_lost input.
    pub offboard_control_signal_lost: bool,
    /// Local receive timestamp in an explicitly named receiver clock domain.
    pub received_at: TimestampV1,
    /// Domain-separated commitment to all fields above.
    pub observation_digest_hex: String,
}

impl Px4OffboardLossFlagObservationV1 {
    /// Construct and content-bind one captured Offboard loss-flag observation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        controller_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_sequence: u64,
        source_timestamp_us: u64,
        mode_req_offboard_signal_raw: u32,
        offboard_control_signal_lost: bool,
        received_at: TimestampV1,
    ) -> Result<Self, Px4OffboardLossFlagError> {
        let mut value = Self {
            schema_version: PX4_OFFBOARD_LOSS_FLAG_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            controller_profile_id: controller_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_sequence,
            source_timestamp_us,
            mode_req_offboard_signal_raw,
            offboard_control_signal_lost,
            received_at,
            observation_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.observation_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate identifiers, receiver timestamp, schema, and content commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardLossFlagError> {
        self.validate_without_digest()?;
        if self.observation_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardLossFlagError::DigestMismatch("observation"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact observation record.
    pub fn observation_id(&self) -> Result<String, Px4OffboardLossFlagError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.offboard-loss-flag.v1:{}",
            self.observation_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardLossFlagError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.controller_profile_id, "controller_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        self.received_at
            .validate()
            .map_err(Px4OffboardLossFlagError::Timestamp)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.controller_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        hasher.update(&self.capture_sequence.to_le_bytes());
        hasher.update(&self.source_timestamp_us.to_le_bytes());
        hasher.update(&self.mode_req_offboard_signal_raw.to_le_bytes());
        hasher.update(&[u8::from(self.offboard_control_signal_lost)]);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Exact provenance and receiver-clock profile for a loss-flag observation window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardLossFlagProfileV1 {
    /// Profile schema version.
    pub schema_version: u16,
    /// Stable versioned policy identity.
    pub profile_id: String,
    /// Bound vehicle/autopilot instance identity.
    pub target_instance_id: String,
    /// Declared controller firmware, vehicle class, and FailsafeFlags schema profile.
    pub controller_profile_id: String,
    /// Expected decoder/telemetry producer identity.
    pub producer_profile_id: String,
    /// Expected PX4 boot/session identity.
    pub source_boot_session_id: String,
    /// Clock in which received timestamps and freshness are evaluated.
    pub receiver_clock_domain: ClockDomainId,
    /// Minimum repeated observations required to report a transition.
    pub min_samples: u16,
    /// Maximum age of the newest local receive timestamp.
    pub max_latest_observation_age_ns: u64,
    /// Domain-separated profile commitment.
    pub profile_digest_hex: String,
}

impl Px4OffboardLossFlagProfileV1 {
    /// Construct and content-bind a versioned loss-flag observation profile.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        controller_profile_id: impl Into<String>,
        producer_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        receiver_clock_domain: ClockDomainId,
        min_samples: u16,
        max_latest_observation_age_ns: u64,
    ) -> Result<Self, Px4OffboardLossFlagError> {
        let mut value = Self {
            schema_version: PX4_OFFBOARD_LOSS_FLAG_SCHEMA_V1,
            profile_id: profile_id.into(),
            target_instance_id: target_instance_id.into(),
            controller_profile_id: controller_profile_id.into(),
            producer_profile_id: producer_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            receiver_clock_domain,
            min_samples,
            max_latest_observation_age_ns,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate profile values and the stored profile commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardLossFlagError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardLossFlagError::DigestMismatch("profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardLossFlagError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.controller_profile_id, "controller_profile_id")?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        self.receiver_clock_domain
            .validate()
            .map_err(Px4OffboardLossFlagError::Timestamp)?;
        if self.min_samples < 2 {
            return Err(Px4OffboardLossFlagError::MinimumSamplesTooSmall);
        }
        if self.max_latest_observation_age_ns == 0 {
            return Err(Px4OffboardLossFlagError::ZeroLatestAge);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.controller_profile_id);
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        feed_str(&mut hasher, self.receiver_clock_domain.as_str());
        hasher.update(&self.min_samples.to_le_bytes());
        hasher.update(&self.max_latest_observation_age_ns.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Outcome of inspecting the latest Offboard-loss input flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Px4OffboardLossFlagOutcomeV1 {
    /// No record was available or the sample floor was not met.
    Incomplete {
        /// Number of retained records.
        available_samples: usize,
        /// Number required by the profile.
        required_samples: u16,
        /// Latest decoded flag, if present; not a qualified current-state assertion.
        latest_loss_flag: Option<bool>,
    },
    /// The latest record exceeded the receiver-clock age ceiling.
    Stale {
        /// Measured receiver-clock age.
        age_ns: u64,
        /// Profile age ceiling.
        max_age_ns: u64,
        /// Last decoded flag, retained only as a stale historical value.
        last_observed_loss_flag: bool,
    },
    /// Repeated fresh records support reporting the latest flag and its last transition.
    FreshObservation {
        /// Current decoded FailsafeFlags offboard_control_signal_lost value.
        offboard_control_signal_lost: bool,
        /// Previous decoded flag value, if available.
        previous_loss_flag: Option<bool>,
        /// Whether the decoded loss flag changed across the last two records.
        loss_transition_observed: bool,
        /// Latest raw mode_req_offboard_signal field, uninterpreted.
        mode_req_offboard_signal_raw: u32,
        /// Age of the newest receiver-side observation.
        latest_observation_age_ns: u64,
        /// Content-addressed ID of the latest source record.
        latest_observation_id: String,
    },
}

/// Content-addressed assessment of one repeated Offboard-loss-flag window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardLossFlagAssessmentV1 {
    /// Assessment schema version.
    pub schema_version: u16,
    /// Exact provenance/freshness profile.
    pub profile: Px4OffboardLossFlagProfileV1,
    /// Receiver-local time at which the window was assessed.
    pub assessed_at: TimestampV1,
    /// Captured source records ordered by receiver timestamp.
    pub samples: Vec<Px4OffboardLossFlagObservationV1>,
    /// Domain-separated assessment commitment.
    pub assessment_digest_hex: String,
}

impl Px4OffboardLossFlagAssessmentV1 {
    /// Construct, sort, bind, and validate a loss-flag window.
    pub fn new(
        profile: Px4OffboardLossFlagProfileV1,
        assessed_at: TimestampV1,
        mut samples: Vec<Px4OffboardLossFlagObservationV1>,
    ) -> Result<Self, Px4OffboardLossFlagError> {
        samples.sort_by_key(|sample| sample.received_at.nanoseconds);
        let mut value = Self {
            schema_version: PX4_OFFBOARD_LOSS_FLAG_SCHEMA_V1,
            profile,
            assessed_at,
            samples,
            assessment_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.assessment_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate all nested records and the assessment commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardLossFlagError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardLossFlagError::DigestMismatch("assessment"));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact observation window.
    pub fn assessment_id(&self) -> Result<String, Px4OffboardLossFlagError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.offboard-loss-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Report the newest flag without conflating it with VehicleStatus.failsafe.
    pub fn outcome(&self) -> Result<Px4OffboardLossFlagOutcomeV1, Px4OffboardLossFlagError> {
        self.validate()?;
        let Some(latest) = self.samples.last() else {
            return Ok(Px4OffboardLossFlagOutcomeV1::Incomplete {
                available_samples: 0,
                required_samples: self.profile.min_samples,
                latest_loss_flag: None,
            });
        };
        let age_ns = self
            .assessed_at
            .elapsed_since(&latest.received_at)
            .map_err(Px4OffboardLossFlagError::Timestamp)?;
        if age_ns > self.profile.max_latest_observation_age_ns {
            return Ok(Px4OffboardLossFlagOutcomeV1::Stale {
                age_ns,
                max_age_ns: self.profile.max_latest_observation_age_ns,
                last_observed_loss_flag: latest.offboard_control_signal_lost,
            });
        }
        if self.samples.len() < usize::from(self.profile.min_samples) {
            return Ok(Px4OffboardLossFlagOutcomeV1::Incomplete {
                available_samples: self.samples.len(),
                required_samples: self.profile.min_samples,
                latest_loss_flag: Some(latest.offboard_control_signal_lost),
            });
        }
        let previous = self.samples.get(self.samples.len() - 2);
        Ok(Px4OffboardLossFlagOutcomeV1::FreshObservation {
            offboard_control_signal_lost: latest.offboard_control_signal_lost,
            previous_loss_flag: previous.map(|sample| sample.offboard_control_signal_lost),
            loss_transition_observed: previous.is_some_and(|sample| {
                sample.offboard_control_signal_lost != latest.offboard_control_signal_lost
            }),
            mode_req_offboard_signal_raw: latest.mode_req_offboard_signal_raw,
            latest_observation_age_ns: age_ns,
            latest_observation_id: latest.observation_id()?,
        })
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardLossFlagError> {
        validate_schema(self.schema_version)?;
        self.profile.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4OffboardLossFlagError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4OffboardLossFlagError::AssessmentClockMismatch);
        }
        let mut seen_digest = HashSet::with_capacity(self.samples.len());
        let mut seen_sequence = HashSet::with_capacity(self.samples.len());
        let mut previous_receive_ns: Option<u64> = None;
        let mut previous_source_us: Option<u64> = None;
        let mut previous_sequence: Option<u64> = None;
        for sample in &self.samples {
            sample.validate()?;
            if sample.producer_profile_id != self.profile.producer_profile_id {
                return Err(Px4OffboardLossFlagError::ProducerMismatch);
            }
            if sample.target_instance_id != self.profile.target_instance_id {
                return Err(Px4OffboardLossFlagError::TargetMismatch);
            }
            if sample.controller_profile_id != self.profile.controller_profile_id {
                return Err(Px4OffboardLossFlagError::ControllerProfileMismatch);
            }
            if sample.source_boot_session_id != self.profile.source_boot_session_id {
                return Err(Px4OffboardLossFlagError::BootSessionMismatch);
            }
            if sample.received_at.clock_domain != self.profile.receiver_clock_domain {
                return Err(Px4OffboardLossFlagError::ReceiverClockMismatch);
            }
            if !seen_digest.insert(sample.observation_digest_hex.clone()) {
                return Err(Px4OffboardLossFlagError::DuplicateObservation);
            }
            if !seen_sequence.insert(sample.capture_sequence) {
                return Err(Px4OffboardLossFlagError::DuplicateSequence);
            }
            if previous_receive_ns
                .is_some_and(|previous| sample.received_at.nanoseconds <= previous)
            {
                return Err(Px4OffboardLossFlagError::NonAdvancingReceiveTime);
            }
            if previous_source_us.is_some_and(|previous| sample.source_timestamp_us <= previous) {
                return Err(Px4OffboardLossFlagError::NonAdvancingSourceTime);
            }
            if let Some(previous) = previous_sequence {
                let expected = previous
                    .checked_add(1)
                    .ok_or(Px4OffboardLossFlagError::SequenceOverflow)?;
                if sample.capture_sequence != expected {
                    return Err(Px4OffboardLossFlagError::CaptureSequenceGap {
                        previous,
                        found: sample.capture_sequence,
                    });
                }
            }
            previous_receive_ns = Some(sample.received_at.nanoseconds);
            previous_source_us = Some(sample.source_timestamp_us);
            previous_sequence = Some(sample.capture_sequence);
        }
        if let Some(latest) = self.samples.last() {
            self.assessed_at
                .elapsed_since(&latest.received_at)
                .map_err(|error| match error {
                    EvidenceValidationError::NonMonotonicTimestamp => {
                        Px4OffboardLossFlagError::AssessmentBeforeLatestObservation
                    }
                    other => Px4OffboardLossFlagError::Timestamp(other),
                })?;
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        hasher.update(&(self.samples.len() as u64).to_le_bytes());
        for sample in &self.samples {
            feed_str(&mut hasher, &sample.observation_digest_hex);
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4OffboardLossFlagError> {
    if schema_version != PX4_OFFBOARD_LOSS_FLAG_SCHEMA_V1 {
        return Err(Px4OffboardLossFlagError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), Px4OffboardLossFlagError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4OffboardLossFlagError::InvalidIdentifier(field));
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

/// Error validating PX4 Offboard-loss-flag evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4OffboardLossFlagError {
    /// Unsupported schema version.
    #[error("unsupported PX4 Offboard loss-flag schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported schema version encountered.
        found: u16,
    },
    /// Identifier is invalid.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Observations mixed decoder identities.
    #[error("PX4 Offboard loss-flag producer mismatch")]
    ProducerMismatch,
    /// Observation has the wrong bound vehicle identity.
    #[error("PX4 Offboard loss-flag target mismatch")]
    TargetMismatch,
    /// Observation uses another controller/vehicle/schema profile.
    #[error("PX4 Offboard loss-flag controller profile mismatch")]
    ControllerProfileMismatch,
    /// Observation is from a different PX4 reboot/boot-relative time session.
    #[error("PX4 Offboard loss-flag source boot session mismatch")]
    BootSessionMismatch,
    /// Source records use another local receive clock.
    #[error("PX4 Offboard loss-flag receiver clock mismatch")]
    ReceiverClockMismatch,
    /// Assessment clock differs from the profile clock.
    #[error("PX4 Offboard loss-flag assessment clock mismatch")]
    AssessmentClockMismatch,
    /// Duplicate content-addressed record.
    #[error("duplicate PX4 Offboard loss-flag observation")]
    DuplicateObservation,
    /// Capture sequence was reused.
    #[error("duplicate PX4 Offboard loss-flag capture sequence")]
    DuplicateSequence,
    /// At least one captured record was skipped.
    #[error("PX4 Offboard loss-flag capture sequence gap after {previous}: found {found}")]
    CaptureSequenceGap {
        /// Previous captured sequence.
        previous: u64,
        /// Found capture sequence.
        found: u64,
    },
    /// Capture sequence cannot advance past u64::MAX.
    #[error("PX4 Offboard loss-flag capture sequence overflow")]
    SequenceOverflow,
    /// Local receive timestamp did not advance.
    #[error("PX4 Offboard loss-flag receive time does not advance strictly")]
    NonAdvancingReceiveTime,
    /// PX4 source timestamp did not advance inside its boot session.
    #[error("PX4 Offboard loss-flag source time does not advance strictly")]
    NonAdvancingSourceTime,
    /// Assessment precedes the latest captured observation.
    #[error("PX4 Offboard loss-flag assessment precedes latest observation")]
    AssessmentBeforeLatestObservation,
    /// Profile requires at least two observations.
    #[error("PX4 Offboard loss-flag min_samples must be at least 2")]
    MinimumSamplesTooSmall,
    /// Freshness ceiling must be non-zero.
    #[error("PX4 Offboard loss-flag maximum age must be non-zero")]
    ZeroLatestAge,
    /// Nested clock or timestamp validation failed.
    #[error("invalid PX4 Offboard loss-flag timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Stored content no longer matches its commitment.
    #[error("PX4 Offboard loss-flag {0} commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(name: &str) -> ClockDomainId {
        ClockDomainId::new(name).unwrap()
    }

    fn ts(ns: u64) -> TimestampV1 {
        TimestampV1::new(clock("companion.monotonic"), ns)
    }

    fn profile() -> Px4OffboardLossFlagProfileV1 {
        Px4OffboardLossFlagProfileV1::new(
            "px4.failsafe-flags.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.failsafe-flags.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            clock("companion.monotonic"),
            2,
            1_000_000_000,
        )
        .unwrap()
    }

    fn observation(
        sequence: u64,
        source_us: u64,
        lost: bool,
        received_ns: u64,
    ) -> Px4OffboardLossFlagObservationV1 {
        Px4OffboardLossFlagObservationV1::new(
            "px4.failsafe-flags.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.boot.fixture.session-a",
            sequence,
            source_us,
            0x21,
            lost,
            ts(received_ns),
        )
        .unwrap()
    }

    fn assessment(
        samples: Vec<Px4OffboardLossFlagObservationV1>,
        assessed_ns: u64,
    ) -> Px4OffboardLossFlagAssessmentV1 {
        Px4OffboardLossFlagAssessmentV1::new(profile(), ts(assessed_ns), samples).unwrap()
    }

    #[test]
    fn specific_offboard_loss_flag_is_not_the_generic_failsafe_state() {
        let value = assessment(
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(11, 2_000, true, 1_100_000_000),
            ],
            1_200_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::FreshObservation {
                offboard_control_signal_lost: true,
                previous_loss_flag: Some(false),
                loss_transition_observed: true,
                mode_req_offboard_signal_raw: 0x21,
                ..
            }
        ));
    }

    #[test]
    fn a_fresh_false_flag_is_only_the_latest_reported_input_value() {
        let value = assessment(
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(11, 2_000, false, 1_100_000_000),
            ],
            1_200_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::FreshObservation {
                offboard_control_signal_lost: false,
                loss_transition_observed: false,
                ..
            }
        ));
    }

    #[test]
    fn stale_outcome_preserves_but_does_not_promote_the_last_flag() {
        let value = assessment(
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(11, 2_000, true, 1_100_000_000),
            ],
            2_100_000_001,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::Stale {
                age_ns: 1_000_000_001,
                max_age_ns: 1_000_000_000,
                last_observed_loss_flag: true,
            }
        ));
    }

    #[test]
    fn observation_age_boundary_is_inclusive() {
        let boundary = assessment(
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(11, 2_000, false, 1_100_000_000),
            ],
            2_100_000_000,
        );
        assert!(matches!(
            boundary.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::FreshObservation {
                latest_observation_age_ns: 1_000_000_000,
                ..
            }
        ));
        let over = assessment(
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(11, 2_000, false, 1_100_000_000),
            ],
            2_100_000_001,
        );
        assert!(matches!(
            over.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::Stale { .. }
        ));
    }

    #[test]
    fn incomplete_history_keeps_the_latest_decoded_flag_visible() {
        let value = Px4OffboardLossFlagAssessmentV1::new(
            profile(),
            ts(1_100_000_000),
            vec![observation(10, 1_000, true, 1_000_000_000)],
        )
        .unwrap();
        assert_eq!(
            value.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::Incomplete {
                available_samples: 1,
                required_samples: 2,
                latest_loss_flag: Some(true),
            }
        );
    }

    #[test]
    fn sequence_gap_and_source_time_regression_fail_closed() {
        let gap = Px4OffboardLossFlagAssessmentV1::new(
            profile(),
            ts(1_200_000_000),
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(12, 2_000, true, 1_100_000_000),
            ],
        );
        assert_eq!(
            gap.unwrap_err(),
            Px4OffboardLossFlagError::CaptureSequenceGap {
                previous: 10,
                found: 12
            }
        );
        let regressed = Px4OffboardLossFlagObservationV1::new(
            "px4.failsafe-flags.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.boot.fixture.session-a",
            11,
            999,
            0x21,
            true,
            ts(1_100_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardLossFlagAssessmentV1::new(
                profile(),
                ts(1_200_000_000),
                vec![observation(10, 1_000, false, 1_000_000_000), regressed],
            )
            .unwrap_err(),
            Px4OffboardLossFlagError::NonAdvancingSourceTime
        );
    }

    #[test]
    fn rebooted_session_cannot_be_mixed_with_prior_source_clock() {
        let rebooted = Px4OffboardLossFlagObservationV1::new(
            "px4.failsafe-flags.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.boot.fixture.session-b",
            11,
            1,
            0x21,
            true,
            ts(1_100_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardLossFlagAssessmentV1::new(
                profile(),
                ts(1_200_000_000),
                vec![observation(10, 1_000, false, 1_000_000_000), rebooted],
            )
            .unwrap_err(),
            Px4OffboardLossFlagError::BootSessionMismatch
        );
    }

    #[test]
    fn wrong_target_producer_and_receiver_clock_fail_closed() {
        let wrong_target = Px4OffboardLossFlagObservationV1::new(
            "px4.failsafe-flags.decoder.fixture.v1",
            "vehicle.other",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.boot.fixture.session-a",
            10,
            1_000,
            0x21,
            true,
            ts(1_000_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardLossFlagAssessmentV1::new(profile(), ts(1_100_000_000), vec![wrong_target],)
                .unwrap_err(),
            Px4OffboardLossFlagError::TargetMismatch
        );
        let wrong_producer = Px4OffboardLossFlagObservationV1::new(
            "other.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.boot.fixture.session-a",
            10,
            1_000,
            0x21,
            true,
            ts(1_000_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardLossFlagAssessmentV1::new(
                profile(),
                ts(1_100_000_000),
                vec![wrong_producer],
            )
            .unwrap_err(),
            Px4OffboardLossFlagError::ProducerMismatch
        );
        let wrong_clock = Px4OffboardLossFlagObservationV1::new(
            "px4.failsafe-flags.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-schema.fixture.v1",
            "px4.boot.fixture.session-a",
            10,
            1_000,
            0x21,
            true,
            TimestampV1::new(clock("another.clock"), 1_000_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardLossFlagAssessmentV1::new(profile(), ts(1_100_000_000), vec![wrong_clock],)
                .unwrap_err(),
            Px4OffboardLossFlagError::ReceiverClockMismatch
        );
    }

    #[test]
    fn input_observation_order_is_canonicalized_for_assessment_identity() {
        let earlier = observation(10, 1_000, false, 1_000_000_000);
        let later = observation(11, 2_000, true, 1_100_000_000);
        let forward = Px4OffboardLossFlagAssessmentV1::new(
            profile(),
            ts(1_200_000_000),
            vec![earlier.clone(), later.clone()],
        )
        .unwrap();
        let reverse = Px4OffboardLossFlagAssessmentV1::new(
            profile(),
            ts(1_200_000_000),
            vec![later, earlier],
        )
        .unwrap();
        assert_eq!(
            forward.assessment_id().unwrap(),
            reverse.assessment_id().unwrap()
        );
    }

    #[test]
    fn zero_samples_are_incomplete_not_zero_loss() {
        let value = assessment(vec![], 1_000);
        assert_eq!(
            value.outcome().unwrap(),
            Px4OffboardLossFlagOutcomeV1::Incomplete {
                available_samples: 0,
                required_samples: 2,
                latest_loss_flag: None,
            }
        );
    }

    #[test]
    fn serde_mutation_and_profile_change_are_detected() {
        let value = assessment(
            vec![
                observation(10, 1_000, false, 1_000_000_000),
                observation(11, 2_000, true, 1_100_000_000),
            ],
            1_200_000_000,
        );
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4OffboardLossFlagAssessmentV1 = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(
            restored.assessment_id().unwrap(),
            value.assessment_id().unwrap()
        );
        restored.samples[0].offboard_control_signal_lost = true;
        assert!(restored.validate().is_err());
        let mut changed_profile = value.clone();
        changed_profile.profile.max_latest_observation_age_ns += 1;
        assert!(changed_profile.validate().is_err());
    }
}
