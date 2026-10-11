// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence for PX4 external-mode SetpointConfig / SetpointConfigReply exchanges.
//!
//! SetpointConfig was added to let external modes declare their setpoint type and
//! requested timeout. The reply echoes type/source ID and a result, but does not
//! echo timeout_ms. A successful reply therefore records PX4 acceptance of the
//! request, not independent readback that the timeout is active or enforced.
//! This module does not send messages, command a mode, arm a vehicle, or qualify flight.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{
    ClockDomainId, EvidenceValidationError, TimestampV1,
};
use thiserror::Error;

/// Schema version for SetpointConfig exchange evidence.
pub const PX4_EXTERNAL_MODE_SETPOINT_CONFIG_SCHEMA_V1: u16 = 1;
/// PX4 SetpointConfig / SetpointConfigReply message definition version.
pub const PX4_SETPOINT_CONFIG_MESSAGE_VERSION_V0: u32 = 0;
const REQUEST_DOMAIN: &[u8] = b"symthaea.autopilot.px4.external-mode-setpoint-config.request.v1\0";
const REPLY_DOMAIN: &[u8] = b"symthaea.autopilot.px4.external-mode-setpoint-config.reply.v1\0";
const PROFILE_DOMAIN: &[u8] = b"symthaea.autopilot.px4.external-mode-setpoint-config.profile.v1\0";
const ASSESSMENT_DOMAIN: &[u8] =
    b"symthaea.autopilot.px4.external-mode-setpoint-config.assessment.v1\0";

/// PX4 SetpointConfig::TYPE values; unrecognized raw values are retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4ExternalModeSetpointTypeV1 {
    /// Sentinel type; not an actionable setpoint declaration.
    Invalid,
    /// Actuator motors and servos.
    DirectActuators,
    /// Multicopter goto setpoint.
    MulticopterGoto,
    /// Fixed-wing lateral and longitudinal setpoints.
    FixedWingLateralLongitudinal,
    /// TrajectorySetpoint.
    Trajectory,
    /// VehicleRatesSetpoint.
    Rates,
    /// VehicleAttitudeSetpoint.
    Attitude,
    /// Rover position setpoint.
    RoverPosition,
    /// Rover speed and attitude setpoints.
    RoverSpeedAttitude,
    /// Rover speed and rate setpoints.
    RoverSpeedRate,
    /// Rover speed and steering setpoints.
    RoverSpeedSteering,
    /// Rover throttle and attitude setpoints.
    RoverThrottleAttitude,
    /// Rover throttle and rate setpoints.
    RoverThrottleRate,
    /// Rover throttle and steering setpoints.
    RoverThrottleSteering,
    /// Six-degree-of-freedom trajectory setpoint.
    Trajectory6Dof,
    /// Vehicle thrust and torque setpoints.
    ThrustAndTorque,
    /// PositionSetpointTriplet.
    PositionTriplet,
    /// A future or unsupported raw type.
    Other(u16),
}

impl Px4ExternalModeSetpointTypeV1 {
    /// Preserve and interpret a raw SetpointConfig type without discarding unknown values.
    pub const fn from_raw(raw: u16) -> Self {
        match raw {
            0 => Self::Invalid,
            1 => Self::DirectActuators,
            2 => Self::MulticopterGoto,
            3 => Self::FixedWingLateralLongitudinal,
            4 => Self::Trajectory,
            5 => Self::Rates,
            6 => Self::Attitude,
            7 => Self::RoverPosition,
            8 => Self::RoverSpeedAttitude,
            9 => Self::RoverSpeedRate,
            10 => Self::RoverSpeedSteering,
            11 => Self::RoverThrottleAttitude,
            12 => Self::RoverThrottleRate,
            13 => Self::RoverThrottleSteering,
            14 => Self::Trajectory6Dof,
            15 => Self::ThrustAndTorque,
            16 => Self::PositionTriplet,
            other => Self::Other(other),
        }
    }

    /// Return the canonical raw value represented by this enum variant.
    pub const fn as_raw(self) -> u16 {
        match self {
            Self::Invalid => 0,
            Self::DirectActuators => 1,
            Self::MulticopterGoto => 2,
            Self::FixedWingLateralLongitudinal => 3,
            Self::Trajectory => 4,
            Self::Rates => 5,
            Self::Attitude => 6,
            Self::RoverPosition => 7,
            Self::RoverSpeedAttitude => 8,
            Self::RoverSpeedRate => 9,
            Self::RoverSpeedSteering => 10,
            Self::RoverThrottleAttitude => 11,
            Self::RoverThrottleRate => 12,
            Self::RoverThrottleSteering => 13,
            Self::Trajectory6Dof => 14,
            Self::ThrustAndTorque => 15,
            Self::PositionTriplet => 16,
            Self::Other(raw) => raw,
        }
    }

    /// Whether this raw value is one of the currently documented concrete setpoint types.
    pub const fn is_known_concrete(self) -> bool {
        matches!(
            self,
            Self::DirectActuators
                | Self::MulticopterGoto
                | Self::FixedWingLateralLongitudinal
                | Self::Trajectory
                | Self::Rates
                | Self::Attitude
                | Self::RoverPosition
                | Self::RoverSpeedAttitude
                | Self::RoverSpeedRate
                | Self::RoverSpeedSteering
                | Self::RoverThrottleAttitude
                | Self::RoverThrottleRate
                | Self::RoverThrottleSteering
                | Self::Trajectory6Dof
                | Self::ThrustAndTorque
                | Self::PositionTriplet
        )
    }
}

/// PX4 SetpointConfigReply::RESULT values, retaining unknown integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4SetpointConfigResultV1 {
    /// PX4 accepted the configuration request.
    Success,
    /// PX4 rejected the request for another reason.
    FailureOther,
    /// The type is unsupported for the current vehicle.
    Unsupported,
    /// PX4 did not recognize the setpoint type.
    UnknownSetpointType,
    /// A future or unsupported raw result.
    Other(u8),
}

impl Px4SetpointConfigResultV1 {
    /// Interpret the raw reply result without discarding unknown values.
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Success,
            1 => Self::FailureOther,
            2 => Self::Unsupported,
            3 => Self::UnknownSetpointType,
            other => Self::Other(other),
        }
    }

    /// Return the canonical raw value represented by this enum variant.
    pub const fn as_raw(self) -> u8 {
        match self {
            Self::Success => 0,
            Self::FailureOther => 1,
            Self::Unsupported => 2,
            Self::UnknownSetpointType => 3,
            Self::Other(raw) => raw,
        }
    }
}

/// One external-mode SetpointConfig request as observed by the adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeSetpointConfigRequestV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// PX4 SetpointConfig message-definition version.
    pub message_version: u32,
    /// Transport/decoder identity.
    pub producer_profile_id: String,
    /// Declared target vehicle / PX4 instance; not authenticated by this record.
    pub target_instance_id: String,
    /// Declared firmware and message-schema profile.
    pub firmware_profile_id: String,
    /// PX4 boot/session identity supplied by the adapter.
    pub source_boot_session_id: String,
    /// Adapter-local ID tying this request/reply pair to one capture attempt.
    pub capture_window_id: String,
    /// PX4 boot-relative SetpointConfig timestamp, microseconds.
    pub source_timestamp_us: u64,
    /// Raw SetpointConfig::type; unknown values are preserved.
    pub type_raw: u16,
    /// Navigation-state ID of the external mode declaring this setpoint type.
    pub source_id: u8,
    /// Whether the mode requested applying this config rather than validation only.
    pub should_apply: bool,
    /// Requested timeout in milliseconds; zero means the timeout is disabled by this request.
    pub timeout_ms: u16,
    /// Companion-monotonic time the adapter submitted this request to PX4.
    pub submitted_at: TimestampV1,
    /// Domain-separated commitment to the exact captured request.
    pub request_digest_hex: String,
}

impl Px4ExternalModeSetpointConfigRequestV1 {
    /// Construct and content-bind one observed SetpointConfig request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_window_id: impl Into<String>,
        source_timestamp_us: u64,
        type_raw: u16,
        source_id: u8,
        should_apply: bool,
        timeout_ms: u16,
        submitted_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeSetpointConfigError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_SETPOINT_CONFIG_SCHEMA_V1,
            message_version: PX4_SETPOINT_CONFIG_MESSAGE_VERSION_V0,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_window_id: capture_window_id.into(),
            source_timestamp_us,
            type_raw,
            source_id,
            should_apply,
            timeout_ms,
            submitted_at,
            request_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.request_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate schema, identities, timestamp, and exact content commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        self.validate_without_digest()?;
        if self.request_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeSetpointConfigError::DigestMismatch("request"));
        }
        Ok(())
    }

    /// Interpreted setpoint type, retaining future values.
    pub const fn setpoint_type(&self) -> Px4ExternalModeSetpointTypeV1 {
        Px4ExternalModeSetpointTypeV1::from_raw(self.type_raw)
    }

    /// Content-addressed identity of this exact request observation.
    pub fn request_id(&self) -> Result<String, Px4ExternalModeSetpointConfigError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.external-mode-setpoint-config.request.v1:{}",
            self.request_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        validate_schema(self.schema_version)?;
        if self.message_version != PX4_SETPOINT_CONFIG_MESSAGE_VERSION_V0 {
            return Err(Px4ExternalModeSetpointConfigError::UnsupportedMessageVersion {
                found: self.message_version,
            });
        }
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        validate_identifier(&self.capture_window_id, "capture_window_id")?;
        self.submitted_at
            .validate()
            .map_err(Px4ExternalModeSetpointConfigError::Timestamp)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REQUEST_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(&self.message_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.firmware_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        feed_str(&mut hasher, &self.capture_window_id);
        hasher.update(&self.source_timestamp_us.to_le_bytes());
        hasher.update(&self.type_raw.to_le_bytes());
        hasher.update(&[self.source_id, u8::from(self.should_apply)]);
        hasher.update(&self.timeout_ms.to_le_bytes());
        feed_str(&mut hasher, self.submitted_at.clock_domain.as_str());
        hasher.update(&self.submitted_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// One PX4 SetpointConfigReply observed by the adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeSetpointConfigReplyV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// PX4 SetpointConfigReply message-definition version.
    pub message_version: u32,
    /// Transport/decoder identity.
    pub producer_profile_id: String,
    /// Declared target vehicle / PX4 instance; not authenticated by this record.
    pub target_instance_id: String,
    /// Declared firmware and message-schema profile.
    pub firmware_profile_id: String,
    /// PX4 boot/session identity supplied by the adapter.
    pub source_boot_session_id: String,
    /// Adapter-local ID tying this reply to one captured request attempt.
    pub capture_window_id: String,
    /// PX4 boot-relative reply timestamp, microseconds.
    pub source_timestamp_us: u64,
    /// Raw SetpointConfigReply::type.
    pub type_raw: u16,
    /// Navigation-state ID copied from the configuration request.
    pub source_id: u8,
    /// Raw SetpointConfigReply::result; future values are preserved.
    pub result_raw: u8,
    /// Mode requirement reported by PX4 for angular velocity.
    pub mode_req_angular_velocity: bool,
    /// Mode requirement reported by PX4 for attitude.
    pub mode_req_attitude: bool,
    /// Mode requirement reported by PX4 for local altitude.
    pub mode_req_local_alt: bool,
    /// Mode requirement reported by PX4 for local position.
    pub mode_req_local_position: bool,
    /// Companion-side receive time, not a PX4 source timestamp.
    pub received_at: TimestampV1,
    /// Domain-separated commitment to the exact captured reply.
    pub reply_digest_hex: String,
}

impl Px4ExternalModeSetpointConfigReplyV1 {
    /// Construct and content-bind one observed SetpointConfigReply.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_window_id: impl Into<String>,
        source_timestamp_us: u64,
        type_raw: u16,
        source_id: u8,
        result_raw: u8,
        mode_req_angular_velocity: bool,
        mode_req_attitude: bool,
        mode_req_local_alt: bool,
        mode_req_local_position: bool,
        received_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeSetpointConfigError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_SETPOINT_CONFIG_SCHEMA_V1,
            message_version: PX4_SETPOINT_CONFIG_MESSAGE_VERSION_V0,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_window_id: capture_window_id.into(),
            source_timestamp_us,
            type_raw,
            source_id,
            result_raw,
            mode_req_angular_velocity,
            mode_req_attitude,
            mode_req_local_alt,
            mode_req_local_position,
            received_at,
            reply_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.reply_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate schema, identities, timestamp, and exact content commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        self.validate_without_digest()?;
        if self.reply_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeSetpointConfigError::DigestMismatch("reply"));
        }
        Ok(())
    }

    /// Interpreted result, retaining future result values.
    pub const fn result(&self) -> Px4SetpointConfigResultV1 {
        Px4SetpointConfigResultV1::from_raw(self.result_raw)
    }

    /// Content-addressed identity of this exact reply observation.
    pub fn reply_id(&self) -> Result<String, Px4ExternalModeSetpointConfigError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.external-mode-setpoint-config.reply.v1:{}",
            self.reply_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        validate_schema(self.schema_version)?;
        if self.message_version != PX4_SETPOINT_CONFIG_MESSAGE_VERSION_V0 {
            return Err(Px4ExternalModeSetpointConfigError::UnsupportedMessageVersion {
                found: self.message_version,
            });
        }
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        validate_identifier(&self.capture_window_id, "capture_window_id")?;
        self.received_at
            .validate()
            .map_err(Px4ExternalModeSetpointConfigError::Timestamp)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPLY_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(&self.message_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.firmware_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        feed_str(&mut hasher, &self.capture_window_id);
        hasher.update(&self.source_timestamp_us.to_le_bytes());
        hasher.update(&self.type_raw.to_le_bytes());
        hasher.update(&[self.source_id, self.result_raw]);
        hasher.update(&[
            u8::from(self.mode_req_angular_velocity),
            u8::from(self.mode_req_attitude),
            u8::from(self.mode_req_local_alt),
            u8::from(self.mode_req_local_position),
        ]);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Provenance and timing bounds for a SetpointConfig request/reply observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeSetpointConfigProfileV1 {
    /// Profile schema version.
    pub schema_version: u16,
    /// Stable profile identity.
    pub profile_id: String,
    /// Expected target vehicle / PX4 instance identity.
    pub target_instance_id: String,
    /// Expected firmware and message-schema profile.
    pub firmware_profile_id: String,
    /// Expected transport/decoder identity.
    pub producer_profile_id: String,
    /// Expected PX4 boot/session identity.
    pub source_boot_session_id: String,
    /// Exact adapter-local request/reply capture window.
    pub capture_window_id: String,
    /// Clock used for receive-time freshness.
    pub receiver_clock_domain: ClockDomainId,
    /// Maximum request-to-reply latency measured in PX4 boot-relative microseconds.
    pub max_source_reply_delay_us: u64,
    /// Time allowed for a request to await a reply in the companion clock.
    pub max_reply_wait_ns: u64,
    /// Maximum age of a received reply in the companion clock.
    pub max_reply_age_ns: u64,
    /// Domain-separated profile commitment.
    pub profile_digest_hex: String,
}

impl Px4ExternalModeSetpointConfigProfileV1 {
    /// Construct and content-bind a profile with explicit timing bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        producer_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_window_id: impl Into<String>,
        receiver_clock_domain: ClockDomainId,
        max_source_reply_delay_us: u64,
        max_reply_wait_ns: u64,
        max_reply_age_ns: u64,
    ) -> Result<Self, Px4ExternalModeSetpointConfigError> {
        let mut profile = Self {
            schema_version: PX4_EXTERNAL_MODE_SETPOINT_CONFIG_SCHEMA_V1,
            profile_id: profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            producer_profile_id: producer_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_window_id: capture_window_id.into(),
            receiver_clock_domain,
            max_source_reply_delay_us,
            max_reply_wait_ns,
            max_reply_age_ns,
            profile_digest_hex: String::new(),
        };
        profile.validate_without_digest()?;
        profile.profile_digest_hex = profile.compute_digest_hex();
        profile.validate()?;
        Ok(profile)
    }

    /// Validate profile values and the exact profile commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeSetpointConfigError::DigestMismatch("profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        validate_identifier(&self.capture_window_id, "capture_window_id")?;
        self.receiver_clock_domain
            .validate()
            .map_err(Px4ExternalModeSetpointConfigError::Timestamp)?;
        if self.max_source_reply_delay_us == 0
            || self.max_reply_wait_ns == 0
            || self.max_reply_age_ns == 0
        {
            return Err(Px4ExternalModeSetpointConfigError::ZeroTimingBound);
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
        feed_str(&mut hasher, &self.capture_window_id);
        feed_str(&mut hasher, self.receiver_clock_domain.as_str());
        hasher.update(&self.max_source_reply_delay_us.to_le_bytes());
        hasher.update(&self.max_reply_wait_ns.to_le_bytes());
        hasher.update(&self.max_reply_age_ns.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Outcome of assessing an external-mode configuration request and its reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Px4ExternalModeSetpointConfigOutcomeV1 {
    /// The companion is still inside the profile's response-wait window.
    AwaitingReply {
        /// Age of the request in the companion clock.
        request_age_ns: u64,
        /// Maximum wait allowed by the profile.
        max_reply_wait_ns: u64,
    },
    /// No reply was observed before the profile's response-wait ceiling.
    ReplyMissing {
        /// Age of the request in the companion clock.
        request_age_ns: u64,
        /// Maximum wait allowed by the profile.
        max_reply_wait_ns: u64,
    },
    /// A reply was received, but is no longer fresh enough to report as current.
    StaleReply {
        /// Age of the reply in the companion clock.
        reply_age_ns: u64,
        /// Maximum permitted reply age.
        max_reply_age_ns: u64,
    },
    /// PX4's boot-relative request/reply delay exceeded its profile bound.
    SourceReplyTooLate {
        /// Measured PX4 source-time latency.
        source_reply_delay_us: u64,
        /// Maximum permitted source-time latency.
        max_source_reply_delay_us: u64,
    },
    /// The reply reported a non-success result; raw value is preserved.
    ReplyRejected {
        /// Raw SetpointConfigReply.result.
        result_raw: u8,
    },
    /// PX4 replied success, but the type is invalid or unrecognized by this schema.
    UnknownSetpointType {
        /// Raw setpoint type preserved without promotion.
        type_raw: u16,
    },
    /// PX4 accepted a check-only request; the current configuration was not requested to change.
    ValidationRequestAccepted {
        /// Raw setpoint type.
        type_raw: u16,
        /// External-mode source/nav-state ID.
        source_id: u8,
        /// Requested timeout; zero means disabled in the request.
        requested_timeout_ms: u16,
        /// Whether the request asks PX4 to configure a timeout.
        timeout_requested_enabled: bool,
        /// Content-addressed request observation.
        request_id: String,
        /// Content-addressed reply observation.
        reply_id: String,
    },
    /// PX4 reported success for a request that asked to apply the configuration.
    ///
    /// The reply does not echo timeout_ms: this records request acceptance, not an
    /// independent readback that the requested timeout is installed or enforced.
    ApplyRequestAccepted {
        /// Raw setpoint type.
        type_raw: u16,
        /// External-mode source/nav-state ID.
        source_id: u8,
        /// Timeout value requested in SetpointConfig.
        requested_timeout_ms: u16,
        /// Whether the request asked PX4 to configure an enabled timeout.
        timeout_requested_enabled: bool,
        /// PX4-reported mode requirement flag.
        mode_req_angular_velocity: bool,
        /// PX4-reported mode requirement flag.
        mode_req_attitude: bool,
        /// PX4-reported mode requirement flag.
        mode_req_local_alt: bool,
        /// PX4-reported mode requirement flag.
        mode_req_local_position: bool,
        /// Content-addressed request observation.
        request_id: String,
        /// Content-addressed reply observation.
        reply_id: String,
    },
}

/// Content-addressed request/reply assessment for one capture window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeSetpointConfigAssessmentV1 {
    /// Assessment schema version.
    pub schema_version: u16,
    /// Exact provenance and timing profile.
    pub profile: Px4ExternalModeSetpointConfigProfileV1,
    /// Captured SetpointConfig request.
    pub request: Px4ExternalModeSetpointConfigRequestV1,
    /// Captured reply, absent until the adapter observes one.
    pub reply: Option<Px4ExternalModeSetpointConfigReplyV1>,
    /// Receiver-clock time at which the pair/window was assessed.
    pub assessed_at: TimestampV1,
    /// Domain-separated assessment commitment.
    pub assessment_digest_hex: String,
}

impl Px4ExternalModeSetpointConfigAssessmentV1 {
    /// Construct, bind, and validate an assessment; a missing reply remains representable.
    pub fn new(
        profile: Px4ExternalModeSetpointConfigProfileV1,
        request: Px4ExternalModeSetpointConfigRequestV1,
        reply: Option<Px4ExternalModeSetpointConfigReplyV1>,
        assessed_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeSetpointConfigError> {
        let mut assessment = Self {
            schema_version: PX4_EXTERNAL_MODE_SETPOINT_CONFIG_SCHEMA_V1,
            profile,
            request,
            reply,
            assessed_at,
            assessment_digest_hex: String::new(),
        };
        assessment.validate_without_digest()?;
        assessment.assessment_digest_hex = assessment.compute_digest_hex();
        assessment.validate()?;
        Ok(assessment)
    }

    /// Validate nested request/reply provenance, chronology, and assessment commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeSetpointConfigError::DigestMismatch("assessment"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact exchange assessment.
    pub fn assessment_id(&self) -> Result<String, Px4ExternalModeSetpointConfigError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.external-mode-setpoint-config.assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Assess reply presence, source-time latency, receive-time freshness, and result.
    pub fn outcome(
        &self,
    ) -> Result<Px4ExternalModeSetpointConfigOutcomeV1, Px4ExternalModeSetpointConfigError> {
        self.validate()?;
        let Some(reply) = &self.reply else {
            let request_age_ns = self
                .assessed_at
                .elapsed_since(&self.request.submitted_at)
                .map_err(Px4ExternalModeSetpointConfigError::Timestamp)?;
            return if request_age_ns > self.profile.max_reply_wait_ns {
                Ok(Px4ExternalModeSetpointConfigOutcomeV1::ReplyMissing {
                    request_age_ns,
                    max_reply_wait_ns: self.profile.max_reply_wait_ns,
                })
            } else {
                Ok(Px4ExternalModeSetpointConfigOutcomeV1::AwaitingReply {
                    request_age_ns,
                    max_reply_wait_ns: self.profile.max_reply_wait_ns,
                })
            };
        };

        let reply_age_ns = self
            .assessed_at
            .elapsed_since(&reply.received_at)
            .map_err(Px4ExternalModeSetpointConfigError::Timestamp)?;
        if reply_age_ns > self.profile.max_reply_age_ns {
            return Ok(Px4ExternalModeSetpointConfigOutcomeV1::StaleReply {
                reply_age_ns,
                max_reply_age_ns: self.profile.max_reply_age_ns,
            });
        }
        let source_reply_delay_us = reply.source_timestamp_us - self.request.source_timestamp_us;
        if source_reply_delay_us > self.profile.max_source_reply_delay_us {
            return Ok(Px4ExternalModeSetpointConfigOutcomeV1::SourceReplyTooLate {
                source_reply_delay_us,
                max_source_reply_delay_us: self.profile.max_source_reply_delay_us,
            });
        }
        if reply.result() != Px4SetpointConfigResultV1::Success {
            return Ok(Px4ExternalModeSetpointConfigOutcomeV1::ReplyRejected {
                result_raw: reply.result_raw,
            });
        }
        if !self.request.setpoint_type().is_known_concrete() {
            return Ok(Px4ExternalModeSetpointConfigOutcomeV1::UnknownSetpointType {
                type_raw: self.request.type_raw,
            });
        }

        let common = (
            self.request.type_raw,
            self.request.source_id,
            self.request.timeout_ms,
            self.request.timeout_ms > 0,
            self.request.request_id()?,
            reply.reply_id()?,
        );
        if !self.request.should_apply {
            return Ok(Px4ExternalModeSetpointConfigOutcomeV1::ValidationRequestAccepted {
                type_raw: common.0,
                source_id: common.1,
                requested_timeout_ms: common.2,
                timeout_requested_enabled: common.3,
                request_id: common.4,
                reply_id: common.5,
            });
        }
        Ok(Px4ExternalModeSetpointConfigOutcomeV1::ApplyRequestAccepted {
            type_raw: common.0,
            source_id: common.1,
            requested_timeout_ms: common.2,
            timeout_requested_enabled: common.3,
            mode_req_angular_velocity: reply.mode_req_angular_velocity,
            mode_req_attitude: reply.mode_req_attitude,
            mode_req_local_alt: reply.mode_req_local_alt,
            mode_req_local_position: reply.mode_req_local_position,
            request_id: common.4,
            reply_id: common.5,
        })
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        validate_schema(self.schema_version)?;
        self.profile.validate()?;
        self.request.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4ExternalModeSetpointConfigError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeSetpointConfigError::AssessmentClockMismatch);
        }
        self.check_request_profile()?;
        self.assessed_at
            .elapsed_since(&self.request.submitted_at)
            .map_err(|error| match error {
                EvidenceValidationError::NonMonotonicTimestamp => {
                    Px4ExternalModeSetpointConfigError::AssessmentBeforeRequest
                }
                other => Px4ExternalModeSetpointConfigError::Timestamp(other),
            })?;
        if let Some(reply) = &self.reply {
            reply.validate()?;
            self.check_reply_profile(reply)?;
            if reply.type_raw != self.request.type_raw {
                return Err(Px4ExternalModeSetpointConfigError::ReplyTypeMismatch);
            }
            if reply.source_id != self.request.source_id {
                return Err(Px4ExternalModeSetpointConfigError::ReplySourceMismatch);
            }
            if reply.source_timestamp_us < self.request.source_timestamp_us {
                return Err(Px4ExternalModeSetpointConfigError::SourceTimeRegression);
            }
            if reply.received_at.nanoseconds < self.request.submitted_at.nanoseconds {
                return Err(Px4ExternalModeSetpointConfigError::ReceiveOrderRegression);
            }
            self.assessed_at
                .elapsed_since(&reply.received_at)
                .map_err(|error| match error {
                    EvidenceValidationError::NonMonotonicTimestamp => {
                        Px4ExternalModeSetpointConfigError::AssessmentBeforeReply
                    }
                    other => Px4ExternalModeSetpointConfigError::Timestamp(other),
                })?;
        }
        Ok(())
    }

    fn check_request_profile(&self) -> Result<(), Px4ExternalModeSetpointConfigError> {
        if self.request.producer_profile_id != self.profile.producer_profile_id {
            return Err(Px4ExternalModeSetpointConfigError::ProducerMismatch);
        }
        if self.request.target_instance_id != self.profile.target_instance_id {
            return Err(Px4ExternalModeSetpointConfigError::TargetMismatch);
        }
        if self.request.firmware_profile_id != self.profile.firmware_profile_id {
            return Err(Px4ExternalModeSetpointConfigError::FirmwareProfileMismatch);
        }
        if self.request.source_boot_session_id != self.profile.source_boot_session_id {
            return Err(Px4ExternalModeSetpointConfigError::BootSessionMismatch);
        }
        if self.request.capture_window_id != self.profile.capture_window_id {
            return Err(Px4ExternalModeSetpointConfigError::CaptureWindowMismatch);
        }
        if self.request.submitted_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeSetpointConfigError::ReceiverClockMismatch);
        }
        Ok(())
    }

    fn check_reply_profile(
        &self,
        reply: &Px4ExternalModeSetpointConfigReplyV1,
    ) -> Result<(), Px4ExternalModeSetpointConfigError> {
        if reply.producer_profile_id != self.profile.producer_profile_id {
            return Err(Px4ExternalModeSetpointConfigError::ProducerMismatch);
        }
        if reply.target_instance_id != self.profile.target_instance_id {
            return Err(Px4ExternalModeSetpointConfigError::TargetMismatch);
        }
        if reply.firmware_profile_id != self.profile.firmware_profile_id {
            return Err(Px4ExternalModeSetpointConfigError::FirmwareProfileMismatch);
        }
        if reply.source_boot_session_id != self.profile.source_boot_session_id {
            return Err(Px4ExternalModeSetpointConfigError::BootSessionMismatch);
        }
        if reply.capture_window_id != self.profile.capture_window_id {
            return Err(Px4ExternalModeSetpointConfigError::CaptureWindowMismatch);
        }
        if reply.received_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeSetpointConfigError::ReceiverClockMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, &self.request.request_digest_hex);
        match &self.reply {
            Some(reply) => {
                hasher.update(&[1]);
                feed_str(&mut hasher, &reply.reply_digest_hex);
            }
            None => {
                hasher.update(&[0]);
            }
        };
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4ExternalModeSetpointConfigError> {
    if schema_version != PX4_EXTERNAL_MODE_SETPOINT_CONFIG_SCHEMA_V1 {
        return Err(Px4ExternalModeSetpointConfigError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4ExternalModeSetpointConfigError> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4ExternalModeSetpointConfigError::InvalidIdentifier(field));
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

/// Error validating or assessing SetpointConfig exchange evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4ExternalModeSetpointConfigError {
    /// Unsupported schema version.
    #[error("unsupported external-mode setpoint configuration schema version {found}")]
    UnsupportedSchemaVersion { found: u16 },
    /// Unsupported PX4 SetpointConfig message-definition version.
    #[error("unsupported SetpointConfig message version {found}")]
    UnsupportedMessageVersion { found: u32 },
    /// Identifier is empty, padded, too long, or contains control characters.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Profile timing ceilings must be non-zero.
    #[error("setpoint configuration timing bounds must be non-zero")]
    ZeroTimingBound,
    /// Request/reply used different capture-window identities.
    #[error("request/reply capture-window mismatch")]
    CaptureWindowMismatch,
    /// Producer identity did not match profile.
    #[error("setpoint configuration producer mismatch")]
    ProducerMismatch,
    /// Target identity did not match profile.
    #[error("setpoint configuration target mismatch")]
    TargetMismatch,
    /// Firmware/message profile did not match.
    #[error("setpoint configuration firmware-profile mismatch")]
    FirmwareProfileMismatch,
    /// Boot-session identities did not match.
    #[error("setpoint configuration boot-session mismatch")]
    BootSessionMismatch,
    /// Receive clock differed from profile.
    #[error("setpoint configuration receiver clock mismatch")]
    ReceiverClockMismatch,
    /// Assessment clock differed from profile.
    #[error("setpoint configuration assessment clock mismatch")]
    AssessmentClockMismatch,
    /// Reply did not echo the requested type.
    #[error("SetpointConfigReply type did not match request")]
    ReplyTypeMismatch,
    /// Reply did not echo the requested source ID.
    #[error("SetpointConfigReply source_id did not match request")]
    ReplySourceMismatch,
    /// PX4 source timestamp regressed across request and reply.
    #[error("SetpointConfigReply source time preceded request source time")]
    SourceTimeRegression,
    /// Reply was received before request within the declared single capture window.
    #[error("SetpointConfigReply receive time preceded request receive time")]
    ReceiveOrderRegression,
    /// Assessment preceded the request receive time.
    #[error("assessment preceded SetpointConfig request")]
    AssessmentBeforeRequest,
    /// Assessment preceded the reply receive time.
    #[error("assessment preceded SetpointConfig reply")]
    AssessmentBeforeReply,
    /// Nested timestamp or clock validation failed.
    #[error("invalid setpoint configuration timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// A committed record, profile, or assessment was mutated.
    #[error("setpoint configuration {0} digest mismatch")]
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

    fn request_at(
        capture_window_id: &str,
        type_raw: u16,
        source_id: u8,
        should_apply: bool,
        timeout_ms: u16,
        source_timestamp_us: u64,
        received_ns: u64,
    ) -> Px4ExternalModeSetpointConfigRequestV1 {
        Px4ExternalModeSetpointConfigRequestV1::new(
            "px4.external-mode.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            capture_window_id,
            source_timestamp_us,
            type_raw,
            source_id,
            should_apply,
            timeout_ms,
            ts(received_ns),
        )
        .unwrap()
    }

    fn request() -> Px4ExternalModeSetpointConfigRequestV1 {
        request_at(
            "setpoint-config.capture.fixture.attempt-1",
            4,
            42,
            true,
            250,
            1_000_000,
            1_000_000_000,
        )
    }

    fn reply_at(
        capture_window_id: &str,
        type_raw: u16,
        source_id: u8,
        result_raw: u8,
        source_timestamp_us: u64,
        received_ns: u64,
    ) -> Px4ExternalModeSetpointConfigReplyV1 {
        Px4ExternalModeSetpointConfigReplyV1::new(
            "px4.external-mode.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            capture_window_id,
            source_timestamp_us,
            type_raw,
            source_id,
            result_raw,
            true,
            true,
            false,
            true,
            ts(received_ns),
        )
        .unwrap()
    }

    fn reply() -> Px4ExternalModeSetpointConfigReplyV1 {
        reply_at(
            "setpoint-config.capture.fixture.attempt-1",
            4,
            42,
            0,
            1_050_000,
            1_100_000_000,
        )
    }

    fn profile() -> Px4ExternalModeSetpointConfigProfileV1 {
        Px4ExternalModeSetpointConfigProfileV1::new(
            "px4.external-mode.setpoint-config.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.external-mode.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            "setpoint-config.capture.fixture.attempt-1",
            clock(),
            100_000,
            200_000_000,
            500_000_000,
        )
        .unwrap()
    }

    fn assessment(
        req: Px4ExternalModeSetpointConfigRequestV1,
        rep: Option<Px4ExternalModeSetpointConfigReplyV1>,
        assessed_ns: u64,
    ) -> Px4ExternalModeSetpointConfigAssessmentV1 {
        Px4ExternalModeSetpointConfigAssessmentV1::new(profile(), req, rep, ts(assessed_ns))
            .unwrap()
    }

    #[test]
    fn setpoint_type_values_match_px4_message_contract_and_unknowns_survive() {
        assert_eq!(
            Px4ExternalModeSetpointTypeV1::from_raw(4),
            Px4ExternalModeSetpointTypeV1::Trajectory
        );
        assert_eq!(
            Px4ExternalModeSetpointTypeV1::from_raw(16),
            Px4ExternalModeSetpointTypeV1::PositionTriplet
        );
        assert_eq!(
            Px4ExternalModeSetpointTypeV1::from_raw(700),
            Px4ExternalModeSetpointTypeV1::Other(700)
        );
        assert_eq!(
            Px4ExternalModeSetpointTypeV1::Other(700).as_raw(),
            700
        );
    }

    #[test]
    fn accepted_apply_request_keeps_timeout_as_requested_not_read_back() {
        let value = assessment(request(), Some(reply()), 1_200_000_000);
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::ApplyRequestAccepted {
                type_raw: 4,
                source_id: 42,
                requested_timeout_ms: 250,
                timeout_requested_enabled: true,
                mode_req_angular_velocity: true,
                mode_req_attitude: true,
                mode_req_local_alt: false,
                mode_req_local_position: true,
                ..
            }
        ));
        assert!(value.assessment_id().unwrap().starts_with(
            "symthaea.autopilot.px4.external-mode-setpoint-config.assessment.v1:"
        ));
    }

    #[test]
    fn check_only_request_is_not_reported_as_applied() {
        let req = request_at(
            "setpoint-config.capture.fixture.attempt-1",
            4,
            42,
            false,
            250,
            1_000_000,
            1_000_000_000,
        );
        assert!(matches!(
            assessment(req, Some(reply()), 1_200_000_000).outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::ValidationRequestAccepted {
                requested_timeout_ms: 250,
                ..
            }
        ));
    }

    #[test]
    fn explicit_zero_timeout_is_preserved_as_disabled_request() {
        let req = request_at(
            "setpoint-config.capture.fixture.attempt-1",
            4,
            42,
            true,
            0,
            1_000_000,
            1_000_000_000,
        );
        assert!(matches!(
            assessment(req, Some(reply()), 1_200_000_000).outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::ApplyRequestAccepted {
                requested_timeout_ms: 0,
                timeout_requested_enabled: false,
                ..
            }
        ));
    }

    #[test]
    fn missing_reply_is_awaiting_then_missing_after_the_declared_deadline() {
        assert!(matches!(
            assessment(request(), None, 1_100_000_000).outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::AwaitingReply {
                request_age_ns: 100_000_000,
                ..
            }
        ));
        assert!(matches!(
            assessment(request(), None, 1_300_000_001).outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::ReplyMissing {
                request_age_ns: 300_000_001,
                ..
            }
        ));
    }

    #[test]
    fn unsupported_result_stays_rejected_with_raw_value() {
        assert_eq!(
            assessment(
                request(),
                Some(reply_at(
                    "setpoint-config.capture.fixture.attempt-1",
                    4,
                    42,
                    2,
                    1_050_000,
                    1_100_000_000,
                )),
                1_200_000_000,
            )
            .outcome()
            .unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::ReplyRejected { result_raw: 2 }
        );
    }

    #[test]
    fn mismatched_type_or_source_id_cannot_be_correlated_as_reply() {
        let wrong_type = Px4ExternalModeSetpointConfigAssessmentV1::new(
            profile(),
            request(),
            Some(reply_at(
                "setpoint-config.capture.fixture.attempt-1",
                5,
                42,
                0,
                1_050_000,
                1_100_000_000,
            )),
            ts(1_200_000_000),
        );
        assert_eq!(
            wrong_type.unwrap_err(),
            Px4ExternalModeSetpointConfigError::ReplyTypeMismatch
        );

        let wrong_source = Px4ExternalModeSetpointConfigAssessmentV1::new(
            profile(),
            request(),
            Some(reply_at(
                "setpoint-config.capture.fixture.attempt-1",
                4,
                43,
                0,
                1_050_000,
                1_100_000_000,
            )),
            ts(1_200_000_000),
        );
        assert_eq!(
            wrong_source.unwrap_err(),
            Px4ExternalModeSetpointConfigError::ReplySourceMismatch
        );
    }

    #[test]
    fn capture_window_and_boot_session_mismatches_fail_closed() {
        let wrong_window = Px4ExternalModeSetpointConfigAssessmentV1::new(
            profile(),
            request(),
            Some(reply_at(
                "setpoint-config.capture.fixture.attempt-2",
                4,
                42,
                0,
                1_050_000,
                1_100_000_000,
            )),
            ts(1_200_000_000),
        );
        assert_eq!(
            wrong_window.unwrap_err(),
            Px4ExternalModeSetpointConfigError::CaptureWindowMismatch
        );

        let wrong_boot = Px4ExternalModeSetpointConfigReplyV1::new(
            "px4.external-mode.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-b",
            "setpoint-config.capture.fixture.attempt-1",
            1_050_000,
            4,
            42,
            0,
            true,
            true,
            false,
            true,
            ts(1_100_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4ExternalModeSetpointConfigAssessmentV1::new(
                profile(),
                request(),
                Some(wrong_boot),
                ts(1_200_000_000),
            )
            .unwrap_err(),
            Px4ExternalModeSetpointConfigError::BootSessionMismatch
        );
    }

    #[test]
    fn source_time_latency_and_receiver_freshness_have_separate_bounds() {
        let source_late = assessment(
            request(),
            Some(reply_at(
                "setpoint-config.capture.fixture.attempt-1",
                4,
                42,
                0,
                1_200_001,
                1_100_000_000,
            )),
            1_200_000_000,
        );
        assert_eq!(
            source_late.outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::SourceReplyTooLate {
                source_reply_delay_us: 200_001,
                max_source_reply_delay_us: 100_000,
            }
        );

        let stale = assessment(request(), Some(reply()), 1_700_000_001);
        assert_eq!(
            stale.outcome().unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::StaleReply {
                reply_age_ns: 600_000_001,
                max_reply_age_ns: 500_000_000,
            }
        );
    }

    #[test]
    fn unknown_setpoint_type_is_not_promoted_even_if_reply_says_success() {
        assert_eq!(
            assessment(
                request_at(
                    "setpoint-config.capture.fixture.attempt-1",
                    700,
                    42,
                    true,
                    250,
                    1_000_000,
                    1_000_000_000,
                ),
                Some(reply_at(
                    "setpoint-config.capture.fixture.attempt-1",
                    700,
                    42,
                    0,
                    1_050_000,
                    1_100_000_000,
                )),
                1_200_000_000,
            )
            .outcome()
            .unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::UnknownSetpointType { type_raw: 700 }
        );

        assert_eq!(
            assessment(
                request_at(
                    "setpoint-config.capture.fixture.attempt-1",
                    0,
                    42,
                    true,
                    250,
                    1_000_000,
                    1_000_000_000,
                ),
                Some(reply_at(
                    "setpoint-config.capture.fixture.attempt-1",
                    0,
                    42,
                    0,
                    1_050_000,
                    1_100_000_000,
                )),
                1_200_000_000,
            )
            .outcome()
            .unwrap(),
            Px4ExternalModeSetpointConfigOutcomeV1::UnknownSetpointType { type_raw: 0 }
        );
    }

    #[test]
    fn source_and_receive_clock_ordering_fail_closed() {
        let source_regression = Px4ExternalModeSetpointConfigAssessmentV1::new(
            profile(),
            request(),
            Some(reply_at(
                "setpoint-config.capture.fixture.attempt-1",
                4,
                42,
                0,
                999_999,
                1_100_000_000,
            )),
            ts(1_200_000_000),
        );
        assert_eq!(
            source_regression.unwrap_err(),
            Px4ExternalModeSetpointConfigError::SourceTimeRegression
        );

        let receive_regression = Px4ExternalModeSetpointConfigAssessmentV1::new(
            profile(),
            request(),
            Some(reply_at(
                "setpoint-config.capture.fixture.attempt-1",
                4,
                42,
                0,
                1_050_000,
                999_999_999,
            )),
            ts(1_200_000_000),
        );
        assert_eq!(
            receive_regression.unwrap_err(),
            Px4ExternalModeSetpointConfigError::ReceiveOrderRegression
        );
    }

    #[test]
    fn unsupported_message_version_and_zero_timing_bounds_are_rejected() {
        let mut value = request();
        value.message_version = 1;
        assert_eq!(
            value.validate(),
            Err(Px4ExternalModeSetpointConfigError::UnsupportedMessageVersion {
                found: 1,
            })
        );
        assert_eq!(
            Px4ExternalModeSetpointConfigProfileV1::new(
                "px4.external-mode.invalid-profile.v1",
                "vehicle.slot-a",
                "px4.v1.18.external-mode.fixture.v1",
                "px4.external-mode.decoder.fixture.v1",
                "px4.boot.fixture.session-a",
                "setpoint-config.capture.fixture.attempt-1",
                clock(),
                0,
                200_000_000,
                500_000_000,
            )
            .unwrap_err(),
            Px4ExternalModeSetpointConfigError::ZeroTimingBound
        );
    }

    #[test]
    fn serde_mutation_breaks_request_and_assessment_commitments() {
        let value = assessment(request(), Some(reply()), 1_200_000_000);
        let encoded = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4ExternalModeSetpointConfigAssessmentV1 =
            serde_json::from_slice(&encoded).unwrap();
        restored.validate().unwrap();
        restored.request.timeout_ms = 1;
        assert!(restored.validate().is_err());
    }
}
