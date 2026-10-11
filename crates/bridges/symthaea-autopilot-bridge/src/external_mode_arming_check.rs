// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence that a PX4 external mode's declared requirements agree with its
//! registered component's reply to a particular ArmingCheckRequest.
//!
//! SetpointConfigReply::source_id is a navigation-state/mode ID, while
//! ArmingCheckReply::registration_id is an external-component registration ID.
//! The relationship is made explicit by RegisterExtComponentReply evidence;
//! the numeric IDs are not presumed equal. Agreement does not imply that the
//! component can arm, that the mode is safe, or that a failsafe will execute.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{
    ClockDomainId, EvidenceValidationError, TimestampV1,
};
use thiserror::Error;

use super::external_mode_setpoint_config::{
    Px4ExternalModeSetpointConfigAssessmentV1, Px4ExternalModeSetpointConfigError,
    Px4ExternalModeSetpointConfigOutcomeV1,
};

/// Schema version for external-mode arming-check coherence evidence.
pub const PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1: u16 = 1;
const REGISTRATION_DOMAIN: &[u8] =
    b"symthaea.autopilot.px4.external-mode-registration-binding.v1\0";
const REQUEST_DOMAIN: &[u8] = b"symthaea.autopilot.px4.arming-check-request.v1\0";
const REPLY_DOMAIN: &[u8] = b"symthaea.autopilot.px4.arming-check-reply.v1\0";
const PROFILE_DOMAIN: &[u8] = b"symthaea.autopilot.px4.arming-check-coherence-profile.v1\0";
const ASSESSMENT_DOMAIN: &[u8] =
    b"symthaea.autopilot.px4.arming-check-coherence-assessment.v1\0";

/// Successful or failed external-component registration as observed from PX4.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeRegistrationBindingV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// Registration transport/decoder identity.
    pub producer_profile_id: String,
    /// Declared vehicle/PX4 identity; this record does not authenticate it.
    pub target_instance_id: String,
    /// Firmware and external-mode API profile.
    pub firmware_profile_id: String,
    /// PX4 boot/session identity supplied by the adapter.
    pub source_boot_session_id: String,
    /// Raw RegisterExtComponentReply timestamp, microseconds since PX4 boot.
    pub source_timestamp_us_raw: u64,
    /// Registration request ID echoed by RegisterExtComponentReply.
    pub registration_request_id: u64,
    /// Registered component name as observed.
    pub component_name: String,
    /// PX4-reported ROS 2 API version.
    pub px4_ros2_api_version: u16,
    /// Whether PX4 reported successful registration.
    pub success: bool,
    /// Raw arming-check registration ID; -1 represents PX4's invalid sentinel.
    pub arming_check_id_raw: i8,
    /// Raw navigation-state/mode ID; -1 represents PX4's invalid sentinel.
    pub mode_id_raw: i8,
    /// Raw mode-executor ID; -1 represents PX4's invalid sentinel.
    pub mode_executor_id_raw: i8,
    /// Whether PX4 says the registered mode cannot be selected by the user.
    pub not_user_selectable: bool,
    /// Companion-clock time when the registration reply was observed.
    pub received_at: TimestampV1,
    /// Domain-separated commitment to all preceding registration evidence.
    pub registration_digest_hex: String,
}

impl Px4ExternalModeRegistrationBindingV1 {
    /// Construct a digest-bound observation of RegisterExtComponentReply.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        source_timestamp_us_raw: u64,
        registration_request_id: u64,
        component_name: impl Into<String>,
        px4_ros2_api_version: u16,
        success: bool,
        arming_check_id_raw: i8,
        mode_id_raw: i8,
        mode_executor_id_raw: i8,
        not_user_selectable: bool,
        received_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeArmingCheckError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            source_timestamp_us_raw,
            registration_request_id,
            component_name: component_name.into(),
            px4_ros2_api_version,
            success,
            arming_check_id_raw,
            mode_id_raw,
            mode_executor_id_raw,
            not_user_selectable,
            received_at,
            registration_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.registration_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate registration identity, IDs, timestamp, and content commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        self.validate_without_digest()?;
        if self.registration_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeArmingCheckError::DigestMismatch("registration"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact registration reply.
    pub fn registration_id(&self) -> Result<String, Px4ExternalModeArmingCheckError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.external-mode-registration-binding.v1:{}",
            self.registration_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        validate_identifier(&self.component_name, "component_name")?;
        self.received_at
            .validate()
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if self.arming_check_id_raw < -1
            || self.mode_id_raw < -1
            || self.mode_executor_id_raw < -1
        {
            return Err(Px4ExternalModeArmingCheckError::InvalidRegistrationSentinel);
        }
        if self.success && (self.arming_check_id_raw < 0 || self.mode_id_raw < 0) {
            return Err(Px4ExternalModeArmingCheckError::SuccessfulRegistrationHasInvalidIds);
        }
        if self.arming_check_id_raw >= 32 {
            return Err(Px4ExternalModeArmingCheckError::RegistrationIdOutsideMask);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REGISTRATION_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.firmware_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        hasher.update(&self.source_timestamp_us_raw.to_le_bytes());
        hasher.update(&self.registration_request_id.to_le_bytes());
        feed_str(&mut hasher, &self.component_name);
        hasher.update(&self.px4_ros2_api_version.to_le_bytes());
        hasher.update(&[u8::from(self.success)]);
        hasher.update(&self.arming_check_id_raw.to_le_bytes());
        hasher.update(&self.mode_id_raw.to_le_bytes());
        hasher.update(&self.mode_executor_id_raw.to_le_bytes());
        hasher.update(&[u8::from(self.not_user_selectable)]);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// A PX4 ArmingCheckRequest observed by the external-mode adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeArmingCheckRequestV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// ArmingCheckRequest message version.
    pub message_version: u32,
    /// Transport/decoder identity.
    pub producer_profile_id: String,
    /// Declared target vehicle/PX4 identity; unauthenticated by this evidence.
    pub target_instance_id: String,
    /// Firmware and message schema profile.
    pub firmware_profile_id: String,
    /// PX4 boot/session identity supplied by the adapter.
    pub source_boot_session_id: String,
    /// Adapter-local ID for this particular request/reply capture cycle.
    pub capture_window_id: String,
    /// Raw PX4 request timestamp; retained without cross-clock conversion.
    pub source_timestamp_us_raw: u64,
    /// PX4's request ID used to correlate associated ArmingCheckReply messages.
    pub request_id: u8,
    /// PX4's registered-component mask; a cleared bit may indicate an unresponsive registration.
    pub valid_registrations_mask: u32,
    /// Companion-clock time this request was received by the adapter.
    pub received_at: TimestampV1,
    /// Content commitment.
    pub request_digest_hex: String,
}

impl Px4ExternalModeArmingCheckRequestV1 {
    /// Construct and content-bind one observed ArmingCheckRequest.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_window_id: impl Into<String>,
        source_timestamp_us_raw: u64,
        request_id: u8,
        valid_registrations_mask: u32,
        received_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeArmingCheckError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1,
            message_version: 1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_window_id: capture_window_id.into(),
            source_timestamp_us_raw,
            request_id,
            valid_registrations_mask,
            received_at,
            request_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.request_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate schema, IDs, time evidence, and content commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        self.validate_without_digest()?;
        if self.request_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeArmingCheckError::DigestMismatch("arming-check request"));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact arming-check request.
    pub fn request_evidence_id(&self) -> Result<String, Px4ExternalModeArmingCheckError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.arming-check-request.v1:{}",
            self.request_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        validate_schema(self.schema_version)?;
        if self.message_version != 1 {
            return Err(Px4ExternalModeArmingCheckError::UnsupportedMessageVersion {
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
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
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
        hasher.update(&self.source_timestamp_us_raw.to_le_bytes());
        hasher.update(&[self.request_id]);
        hasher.update(&self.valid_registrations_mask.to_le_bytes());
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// The requirement fields returned by ArmingCheckReply that can be compared
/// with the four requirement fields in SetpointConfigReply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ComparableModeRequirementsV1 {
    /// Requires angular-velocity estimation.
    pub angular_velocity: bool,
    /// Requires attitude estimation.
    pub attitude: bool,
    /// Requires local-altitude estimation.
    pub local_altitude: bool,
    /// Requires local-position estimation.
    pub local_position: bool,
}

impl Px4ComparableModeRequirementsV1 {
    fn from_setpoint_reply(
        value: &super::external_mode_setpoint_config::Px4ExternalModeSetpointConfigReplyV1,
    ) -> Self {
        Self {
            angular_velocity: value.mode_req_angular_velocity,
            attitude: value.mode_req_attitude,
            local_altitude: value.mode_req_local_alt,
            local_position: value.mode_req_local_position,
        }
    }

    fn from_arming_reply(value: &Px4ExternalModeArmingCheckReplyV1) -> Self {
        Self {
            angular_velocity: value.mode_req_angular_velocity,
            attitude: value.mode_req_attitude,
            local_altitude: value.mode_req_local_alt,
            local_position: value.mode_req_local_position,
        }
    }

    fn differences(self, observed: Self) -> Vec<Px4ModeRequirementFieldV1> {
        let mut fields = Vec::with_capacity(4);
        if self.angular_velocity != observed.angular_velocity {
            fields.push(Px4ModeRequirementFieldV1::AngularVelocity);
        }
        if self.attitude != observed.attitude {
            fields.push(Px4ModeRequirementFieldV1::Attitude);
        }
        if self.local_altitude != observed.local_altitude {
            fields.push(Px4ModeRequirementFieldV1::LocalAltitude);
        }
        if self.local_position != observed.local_position {
            fields.push(Px4ModeRequirementFieldV1::LocalPosition);
        }
        fields
    }
}

/// A comparable requirement field that differed between PX4 message reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4ModeRequirementFieldV1 {
    /// Angular-velocity requirement.
    AngularVelocity,
    /// Attitude requirement.
    Attitude,
    /// Local-altitude requirement.
    LocalAltitude,
    /// Local-position requirement.
    LocalPosition,
}

/// One raw PX4 Event entry carried by ArmingCheckReply.
///
/// The arguments are retained as bytes because their interpretation depends on the
/// firmware's event metadata; this adapter does not invent a decoded meaning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeArmingCheckEventV1 {
    /// Raw PX4 event timestamp, microseconds since PX4 system start.
    pub timestamp_us_raw: u64,
    /// Raw PX4 event identifier.
    pub id: u32,
    /// Raw event sequence number.
    pub event_sequence: u16,
    /// Fixed-width, uninterpreted event arguments exactly as captured.
    pub arguments: [u8; 25],
    /// Raw PX4 log-level bit field.
    pub log_levels: u8,
}

/// Fixed number of Event slots in PX4 ArmingCheckReply v1.
pub const PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1: usize = 5;

/// One ArmingCheckReply with its component registration and complete reported requirements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeArmingCheckReplyV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// ArmingCheckReply message version.
    pub message_version: u32,
    /// Transport/decoder identity.
    pub producer_profile_id: String,
    /// Declared target vehicle/PX4 identity; unauthenticated by this evidence.
    pub target_instance_id: String,
    /// Firmware and message schema profile.
    pub firmware_profile_id: String,
    /// PX4 boot/session identity supplied by the adapter.
    pub source_boot_session_id: String,
    /// Adapter-local request/reply capture cycle.
    pub capture_window_id: String,
    /// Raw PX4 reply timestamp; preserved without cross-clock conversion.
    pub source_timestamp_us_raw: u64,
    /// ArmingCheckRequest ID that this reply answers.
    pub request_id: u8,
    /// Registered external-component ID, distinct from navigation-state/mode ID.
    pub registration_id: u8,
    /// Raw health-component index; zero is PX4's HEALTH_COMPONENT_INDEX_NONE sentinel.
    pub health_component_index: u8,
    /// Raw health-component presence flag (currently unused by PX4's events interface).
    pub health_component_is_present: bool,
    /// Raw health-component warning flag (currently unused by PX4's events interface).
    pub health_component_warning: bool,
    /// Raw health-component error flag (currently unused by PX4's events interface).
    pub health_component_error: bool,
    /// Whether this component reports that it can arm/run in its mode.
    pub can_arm_and_run: bool,
    /// Number of valid entries in the fixed-size PX4 Event queue.
    pub num_events: u8,
    /// All five raw Event slots, including unused tail slots, as observed.
    pub events: [Px4ExternalModeArmingCheckEventV1; PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1],
    /// Full requirement vector reported in ArmingCheckReply.
    pub mode_req_angular_velocity: bool,
    /// Attitude requirement reported by the component.
    pub mode_req_attitude: bool,
    /// Local-altitude requirement reported by the component.
    pub mode_req_local_alt: bool,
    /// Local-position requirement reported by the component.
    pub mode_req_local_position: bool,
    /// Relaxed local-position requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_local_position_relaxed: bool,
    /// Global-position requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_global_position: bool,
    /// Relaxed global-position requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_global_position_relaxed: bool,
    /// Mission requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_mission: bool,
    /// Home-position requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_home_position: bool,
    /// Prevent-arming requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_prevent_arming: bool,
    /// Manual-control requirement, preserved but not compared to SetpointConfigReply.
    pub mode_req_manual_control: bool,
    /// Companion-clock time the reply was received.
    pub received_at: TimestampV1,
    /// Content commitment.
    pub reply_digest_hex: String,
}

impl Px4ExternalModeArmingCheckReplyV1 {
    /// Construct and content-bind one observed ArmingCheckReply.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_window_id: impl Into<String>,
        source_timestamp_us_raw: u64,
        request_id: u8,
        registration_id: u8,
        health_component_index: u8,
        health_component_is_present: bool,
        health_component_warning: bool,
        health_component_error: bool,
        can_arm_and_run: bool,
        num_events: u8,
        events: [Px4ExternalModeArmingCheckEventV1; PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1],
        mode_req_angular_velocity: bool,
        mode_req_attitude: bool,
        mode_req_local_alt: bool,
        mode_req_local_position: bool,
        mode_req_local_position_relaxed: bool,
        mode_req_global_position: bool,
        mode_req_global_position_relaxed: bool,
        mode_req_mission: bool,
        mode_req_home_position: bool,
        mode_req_prevent_arming: bool,
        mode_req_manual_control: bool,
        received_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeArmingCheckError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1,
            message_version: 1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_window_id: capture_window_id.into(),
            source_timestamp_us_raw,
            request_id,
            registration_id,
            health_component_index,
            health_component_is_present,
            health_component_warning,
            health_component_error,
            can_arm_and_run,
            num_events,
            events,
            mode_req_angular_velocity,
            mode_req_attitude,
            mode_req_local_alt,
            mode_req_local_position,
            mode_req_local_position_relaxed,
            mode_req_global_position,
            mode_req_global_position_relaxed,
            mode_req_mission,
            mode_req_home_position,
            mode_req_prevent_arming,
            mode_req_manual_control,
            received_at,
            reply_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.reply_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate schema, identity, timestamp, and exact content commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        self.validate_without_digest()?;
        if self.reply_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeArmingCheckError::DigestMismatch("arming-check reply"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact arming-check reply.
    pub fn reply_evidence_id(&self) -> Result<String, Px4ExternalModeArmingCheckError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.arming-check-reply.v1:{}",
            self.reply_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        validate_schema(self.schema_version)?;
        if self.message_version != 1 {
            return Err(Px4ExternalModeArmingCheckError::UnsupportedMessageVersion {
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
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if usize::from(self.num_events) > PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1 {
            return Err(Px4ExternalModeArmingCheckError::InvalidEventCount {
                found: self.num_events,
            });
        }
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
        hasher.update(&self.source_timestamp_us_raw.to_le_bytes());
        hasher.update(&[
            self.request_id,
            self.registration_id,
            self.health_component_index,
            u8::from(self.health_component_is_present),
            u8::from(self.health_component_warning),
            u8::from(self.health_component_error),
            u8::from(self.can_arm_and_run),
            self.num_events,
        ]);
        for event in &self.events {
            hasher.update(&event.timestamp_us_raw.to_le_bytes());
            hasher.update(&event.id.to_le_bytes());
            hasher.update(&event.event_sequence.to_le_bytes());
            hasher.update(&event.arguments);
            hasher.update(&[event.log_levels]);
        }
        hasher.update(&[
            u8::from(self.mode_req_angular_velocity),
            u8::from(self.mode_req_attitude),
            u8::from(self.mode_req_local_alt),
            u8::from(self.mode_req_local_position),
            u8::from(self.mode_req_local_position_relaxed),
            u8::from(self.mode_req_global_position),
            u8::from(self.mode_req_global_position_relaxed),
            u8::from(self.mode_req_mission),
            u8::from(self.mode_req_home_position),
            u8::from(self.mode_req_prevent_arming),
            u8::from(self.mode_req_manual_control),
        ]);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Profile for correlating registration, setpoint declaration, and arming check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeArmingCheckProfileV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// Stable profile identity.
    pub profile_id: String,
    /// Expected target vehicle/PX4 identity.
    pub target_instance_id: String,
    /// Expected firmware/API profile.
    pub firmware_profile_id: String,
    /// Expected registration-response decoder identity.
    pub registration_producer_profile_id: String,
    /// Expected ArmingCheckRequest/Reply decoder identity.
    pub arming_check_producer_profile_id: String,
    /// Expected PX4 boot/session identity.
    pub source_boot_session_id: String,
    /// Arming-check request/reply capture cycle.
    pub arming_check_capture_window_id: String,
    /// Clock used to compare local receive/submission events.
    pub receiver_clock_domain: ClockDomainId,
    /// Maximum age of successful registration evidence.
    pub max_registration_age_ns: u64,
    /// Maximum age of the successful SetpointConfig assessment.
    pub max_setpoint_config_age_ns: u64,
    /// Maximum wait for an arming-check reply.
    pub max_reply_wait_ns: u64,
    /// Maximum age of an observed arming-check reply.
    pub max_reply_age_ns: u64,
    /// Domain-separated profile commitment.
    pub profile_digest_hex: String,
}

impl Px4ExternalModeArmingCheckProfileV1 {
    /// Construct an explicit profile with non-zero freshness/latency ceilings.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        firmware_profile_id: impl Into<String>,
        registration_producer_profile_id: impl Into<String>,
        arming_check_producer_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        arming_check_capture_window_id: impl Into<String>,
        receiver_clock_domain: ClockDomainId,
        max_registration_age_ns: u64,
        max_setpoint_config_age_ns: u64,
        max_reply_wait_ns: u64,
        max_reply_age_ns: u64,
    ) -> Result<Self, Px4ExternalModeArmingCheckError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1,
            profile_id: profile_id.into(),
            target_instance_id: target_instance_id.into(),
            firmware_profile_id: firmware_profile_id.into(),
            registration_producer_profile_id: registration_producer_profile_id.into(),
            arming_check_producer_profile_id: arming_check_producer_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            arming_check_capture_window_id: arming_check_capture_window_id.into(),
            receiver_clock_domain,
            max_registration_age_ns,
            max_setpoint_config_age_ns,
            max_reply_wait_ns,
            max_reply_age_ns,
            profile_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.profile_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate profile identity, timing bounds, and commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeArmingCheckError::DigestMismatch("profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.firmware_profile_id, "firmware_profile_id")?;
        validate_identifier(
            &self.registration_producer_profile_id,
            "registration_producer_profile_id",
        )?;
        validate_identifier(
            &self.arming_check_producer_profile_id,
            "arming_check_producer_profile_id",
        )?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        validate_identifier(
            &self.arming_check_capture_window_id,
            "arming_check_capture_window_id",
        )?;
        self.receiver_clock_domain
            .validate()
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if self.max_registration_age_ns == 0
            || self.max_setpoint_config_age_ns == 0
            || self.max_reply_wait_ns == 0
            || self.max_reply_age_ns == 0
        {
            return Err(Px4ExternalModeArmingCheckError::ZeroTimingBound);
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
        feed_str(&mut hasher, &self.registration_producer_profile_id);
        feed_str(&mut hasher, &self.arming_check_producer_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        feed_str(&mut hasher, &self.arming_check_capture_window_id);
        feed_str(&mut hasher, self.receiver_clock_domain.as_str());
        hasher.update(&self.max_registration_age_ns.to_le_bytes());
        hasher.update(&self.max_setpoint_config_age_ns.to_le_bytes());
        hasher.update(&self.max_reply_wait_ns.to_le_bytes());
        hasher.update(&self.max_reply_age_ns.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Configuration gating reason when the preceding SetpointConfig exchange is not eligible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4SetpointConfigBlockReasonV1 {
    /// The exchange has not received a reply yet.
    AwaitingReply,
    /// No correlated reply was captured before its response deadline.
    ReplyNotObservedBeforeDeadline,
    /// The reply is no longer fresh enough.
    StaleReply,
    /// The reply arrived beyond its allowed response time.
    ReplyObservedTooLate,
    /// PX4 rejected the request.
    ReplyRejected { result_raw: u8 },
    /// The request/reply setpoint type is unknown or invalid to this schema.
    UnknownSetpointType { type_raw: u16 },
    /// The request was check-only rather than an apply request.
    CheckOnlyRequest,
}

/// Outcome of correlating SetpointConfig with registration and ArmingCheckReply evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Px4ExternalModeArmingCheckOutcomeV1 {
    /// Required upstream evidence does not qualify a successful apply request.
    SetpointConfigurationNotQualified {
        /// The exact reason the preceding setpoint configuration is not eligible.
        reason: Px4SetpointConfigBlockReasonV1,
    },
    /// Registration failed or the mode and registration IDs could not be established.
    RegistrationNotSuccessful,
    /// Successful registration evidence is too old to bind this observation.
    RegistrationEvidenceStale {
        /// Age of the registration reply.
        registration_age_ns: u64,
        /// Maximum permitted registration age.
        max_registration_age_ns: u64,
    },
    /// The setpoint configuration evidence is too old.
    SetpointConfigurationEvidenceStale {
        /// Age of the configuration assessment.
        configuration_age_ns: u64,
        /// Maximum permitted age.
        max_setpoint_config_age_ns: u64,
    },
    /// Request was received; the response window has not expired.
    AwaitingReply {
        /// Age of the request in the companion clock.
        request_age_ns: u64,
        /// Maximum reply wait.
        max_reply_wait_ns: u64,
    },
    /// No correlated reply was captured by the deadline; cause is not inferred.
    ReplyNotObservedBeforeDeadline {
        /// Age of request in the companion clock.
        request_age_ns: u64,
        /// Maximum reply wait.
        max_reply_wait_ns: u64,
    },
    /// Reply is too old to be considered current.
    StaleReply {
        /// Age of reply in the companion clock.
        reply_age_ns: u64,
        /// Maximum permitted reply age.
        max_reply_age_ns: u64,
    },
    /// Reply arrived later than the request's allowed response window.
    ReplyObservedTooLate {
        /// Time from local request receipt to local reply receipt.
        response_latency_ns: u64,
        /// Maximum permitted reply wait.
        max_reply_wait_ns: u64,
    },
    /// PX4's mask did not mark this registration as valid/responsive for the request.
    RegistrationNotMarkedValid {
        /// Explicit component registration ID.
        registration_id: u8,
        /// Raw request mask.
        valid_registrations_mask: u32,
    },
    /// Requirement flags match; can_arm_and_run remains a separate reported proposition.
    RequirementsMatch {
        /// PX4 registration ID, distinct from the mode/navigation-state ID.
        registration_id: u8,
        /// Mode/navigation-state ID joined through registration evidence.
        mode_id: u8,
        /// Echoed arming-check request ID.
        request_id: u8,
        /// The component's separately reported can-arm/run value.
        can_arm_and_run: bool,
    },
    /// One or more comparable requirement fields disagree.
    RequirementsMismatch {
        /// Deterministically ordered list of differing requirement fields.
        differing_fields: Vec<Px4ModeRequirementFieldV1>,
        /// PX4 registration ID, distinct from the mode/navigation-state ID.
        registration_id: u8,
        /// Mode/navigation-state ID joined through registration evidence.
        mode_id: u8,
        /// Echoed arming-check request ID.
        request_id: u8,
        /// The component's separately reported can-arm/run value.
        can_arm_and_run: bool,
    },
}

/// Content-addressed assessment over one request/reply cycle and its upstream bindings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ExternalModeArmingCheckAssessmentV1 {
    /// Assessment schema version.
    pub schema_version: u16,
    /// Exact provenance and timing profile.
    pub profile: Px4ExternalModeArmingCheckProfileV1,
    /// Explicit mode ID ↔ registered-component ID evidence.
    pub registration: Px4ExternalModeRegistrationBindingV1,
    /// Setpoint configuration request/result that declared the mode requirements.
    pub setpoint_configuration: Px4ExternalModeSetpointConfigAssessmentV1,
    /// PX4 broadcast arming-check request whose ID the reply must echo.
    pub request: Px4ExternalModeArmingCheckRequestV1,
    /// Component reply, absent until the adapter observes one.
    pub reply: Option<Px4ExternalModeArmingCheckReplyV1>,
    /// Companion-clock time when the assessment was made.
    pub assessed_at: TimestampV1,
    /// Domain-separated assessment commitment.
    pub assessment_digest_hex: String,
}

impl Px4ExternalModeArmingCheckAssessmentV1 {
    /// Construct, correlate and digest one arming-check evidence assessment.
    pub fn new(
        profile: Px4ExternalModeArmingCheckProfileV1,
        registration: Px4ExternalModeRegistrationBindingV1,
        setpoint_configuration: Px4ExternalModeSetpointConfigAssessmentV1,
        request: Px4ExternalModeArmingCheckRequestV1,
        reply: Option<Px4ExternalModeArmingCheckReplyV1>,
        assessed_at: TimestampV1,
    ) -> Result<Self, Px4ExternalModeArmingCheckError> {
        let mut item = Self {
            schema_version: PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1,
            profile,
            registration,
            setpoint_configuration,
            request,
            reply,
            assessed_at,
            assessment_digest_hex: String::new(),
        };
        item.validate_without_digest()?;
        item.assessment_digest_hex = item.compute_digest_hex();
        item.validate()?;
        Ok(item)
    }

    /// Validate nested evidence, correlation identities, chronology and commitment.
    pub fn validate(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4ExternalModeArmingCheckError::DigestMismatch("assessment"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact cross-message assessment.
    pub fn assessment_id(&self) -> Result<String, Px4ExternalModeArmingCheckError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.arming-check-coherence-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Assess freshness, upstream configuration qualification, registration validity,
    /// request/reply correlation, and requirement-vector agreement.
    pub fn outcome(
        &self,
    ) -> Result<Px4ExternalModeArmingCheckOutcomeV1, Px4ExternalModeArmingCheckError> {
        self.validate()?;
        let config_age_ns = self
            .assessed_at
            .elapsed_since(&self.setpoint_configuration.assessed_at)
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if config_age_ns > self.profile.max_setpoint_config_age_ns {
            return Ok(Px4ExternalModeArmingCheckOutcomeV1::SetpointConfigurationEvidenceStale {
                configuration_age_ns: config_age_ns,
                max_setpoint_config_age_ns: self.profile.max_setpoint_config_age_ns,
            });
        }
        let registration_age_ns = self
            .assessed_at
            .elapsed_since(&self.registration.received_at)
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if registration_age_ns > self.profile.max_registration_age_ns {
            return Ok(Px4ExternalModeArmingCheckOutcomeV1::RegistrationEvidenceStale {
                registration_age_ns: registration_age_ns,
                max_registration_age_ns: self.profile.max_registration_age_ns,
            });
        }
        let config_outcome = self.setpoint_configuration.outcome()?;
        let is_applied_config = matches!(
            &config_outcome,
            Px4ExternalModeSetpointConfigOutcomeV1::ApplyRequestAccepted { .. }
        );
        if !is_applied_config {
            let reason = match config_outcome {
                Px4ExternalModeSetpointConfigOutcomeV1::AwaitingReply { .. } => {
                    Px4SetpointConfigBlockReasonV1::AwaitingReply
                }
                Px4ExternalModeSetpointConfigOutcomeV1::ReplyNotObservedBeforeDeadline { .. } => {
                    Px4SetpointConfigBlockReasonV1::ReplyNotObservedBeforeDeadline
                }
                Px4ExternalModeSetpointConfigOutcomeV1::StaleReply { .. } => {
                    Px4SetpointConfigBlockReasonV1::StaleReply
                }
                Px4ExternalModeSetpointConfigOutcomeV1::ReplyObservedTooLate { .. } => {
                    Px4SetpointConfigBlockReasonV1::ReplyObservedTooLate
                }
                Px4ExternalModeSetpointConfigOutcomeV1::ReplyRejected { result_raw } => {
                    Px4SetpointConfigBlockReasonV1::ReplyRejected { result_raw }
                }
                Px4ExternalModeSetpointConfigOutcomeV1::UnknownSetpointType { type_raw } => {
                    Px4SetpointConfigBlockReasonV1::UnknownSetpointType { type_raw }
                }
                Px4ExternalModeSetpointConfigOutcomeV1::ValidationRequestAccepted { .. } => {
                    Px4SetpointConfigBlockReasonV1::CheckOnlyRequest
                }
                Px4ExternalModeSetpointConfigOutcomeV1::ApplyRequestAccepted { .. } => {
                    return Err(Px4ExternalModeArmingCheckError::OutcomeClassificationContradiction);
                }
            };
            return Ok(
                Px4ExternalModeArmingCheckOutcomeV1::SetpointConfigurationNotQualified { reason },
            );
        }
        if !self.registration.success
            || self.registration.arming_check_id_raw < 0
            || self.registration.mode_id_raw < 0
        {
            return Ok(Px4ExternalModeArmingCheckOutcomeV1::RegistrationNotSuccessful);
        }
        if (self.request.valid_registrations_mask
            & (1_u32 << (self.registration.arming_check_id_raw as u32)))
            == 0
        {
            return Ok(Px4ExternalModeArmingCheckOutcomeV1::RegistrationNotMarkedValid {
                registration_id: self.registration.arming_check_id_raw as u8,
                valid_registrations_mask: self.request.valid_registrations_mask,
            });
        }
        let Some(reply) = &self.reply else {
            let request_age_ns = self
                .assessed_at
                .elapsed_since(&self.request.received_at)
                .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
            return if request_age_ns > self.profile.max_reply_wait_ns {
                Ok(Px4ExternalModeArmingCheckOutcomeV1::ReplyNotObservedBeforeDeadline {
                    request_age_ns,
                    max_reply_wait_ns: self.profile.max_reply_wait_ns,
                })
            } else {
                Ok(Px4ExternalModeArmingCheckOutcomeV1::AwaitingReply {
                    request_age_ns,
                    max_reply_wait_ns: self.profile.max_reply_wait_ns,
                })
            };
        };
        let reply_age_ns = self
            .assessed_at
            .elapsed_since(&reply.received_at)
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if reply_age_ns > self.profile.max_reply_age_ns {
            return Ok(Px4ExternalModeArmingCheckOutcomeV1::StaleReply {
                reply_age_ns,
                max_reply_age_ns: self.profile.max_reply_age_ns,
            });
        }
        let response_latency_ns = reply
            .received_at
            .elapsed_since(&self.request.received_at)
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if response_latency_ns > self.profile.max_reply_wait_ns {
            return Ok(Px4ExternalModeArmingCheckOutcomeV1::ReplyObservedTooLate {
                response_latency_ns,
                max_reply_wait_ns: self.profile.max_reply_wait_ns,
            });
        }
        let expected = Px4ComparableModeRequirementsV1::from_setpoint_reply(
            self.setpoint_configuration.reply.as_ref().ok_or(
                Px4ExternalModeArmingCheckError::MissingSetpointConfigReply,
            )?,
        );
        let observed = Px4ComparableModeRequirementsV1::from_arming_reply(reply);
        let differing_fields = expected.differences(observed);
        let common = (
            self.registration.arming_check_id_raw as u8,
            self.registration.mode_id_raw as u8,
            self.request.request_id,
            reply.can_arm_and_run,
        );
        if differing_fields.is_empty() {
            Ok(Px4ExternalModeArmingCheckOutcomeV1::RequirementsMatch {
                registration_id: common.0,
                mode_id: common.1,
                request_id: common.2,
                can_arm_and_run: common.3,
            })
        } else {
            Ok(Px4ExternalModeArmingCheckOutcomeV1::RequirementsMismatch {
                differing_fields,
                registration_id: common.0,
                mode_id: common.1,
                request_id: common.2,
                can_arm_and_run: common.3,
            })
        }
    }

    fn validate_without_digest(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        validate_schema(self.schema_version)?;
        self.profile.validate()?;
        self.registration.validate()?;
        self.setpoint_configuration.validate()?;
        self.request.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4ExternalModeArmingCheckError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeArmingCheckError::AssessmentClockMismatch);
        }
        self.check_registration_profile()?;
        self.check_setpoint_profile()?;
        self.check_request_profile()?;
        self.assessed_at
            .elapsed_since(&self.registration.received_at)
            .map_err(map_assessment_timestamp_error)?;
        self.assessed_at
            .elapsed_since(&self.setpoint_configuration.assessed_at)
            .map_err(map_assessment_timestamp_error)?;
        self.assessed_at
            .elapsed_since(&self.request.received_at)
            .map_err(map_assessment_timestamp_error)?;
        if let Some(reply) = &self.reply {
            reply.validate()?;
            self.check_reply_profile(reply)?;
            if reply.request_id != self.request.request_id {
                return Err(Px4ExternalModeArmingCheckError::ReplyRequestIdMismatch);
            }
            if reply.registration_id != self.registration.arming_check_id_raw as u8 {
                return Err(Px4ExternalModeArmingCheckError::ReplyRegistrationIdMismatch);
            }
            // Both raw source timestamps are PX4 boot-relative microseconds. They may
            // be compared to one another only after the boot/session profile has matched.
            if reply.source_timestamp_us_raw < self.request.source_timestamp_us_raw {
                return Err(Px4ExternalModeArmingCheckError::ReplySourceTimestampRegression);
            }
            if reply.received_at.nanoseconds < self.request.received_at.nanoseconds {
                return Err(Px4ExternalModeArmingCheckError::ReceiveOrderRegression);
            }
            self.assessed_at
                .elapsed_since(&reply.received_at)
                .map_err(map_assessment_timestamp_error)?;
        }
        Ok(())
    }

    fn check_registration_profile(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        if self.registration.target_instance_id != self.profile.target_instance_id {
            return Err(Px4ExternalModeArmingCheckError::TargetMismatch);
        }
        if self.registration.firmware_profile_id != self.profile.firmware_profile_id {
            return Err(Px4ExternalModeArmingCheckError::FirmwareProfileMismatch);
        }
        if self.registration.source_boot_session_id != self.profile.source_boot_session_id {
            return Err(Px4ExternalModeArmingCheckError::BootSessionMismatch);
        }
        if self.registration.producer_profile_id != self.profile.registration_producer_profile_id {
            return Err(Px4ExternalModeArmingCheckError::RegistrationProducerMismatch);
        }
        if self.registration.received_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeArmingCheckError::ReceiverClockMismatch);
        }
        Ok(())
    }

    fn check_setpoint_profile(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        let config = &self.setpoint_configuration;
        if config.profile.target_instance_id != self.profile.target_instance_id
            || config.request.target_instance_id != self.profile.target_instance_id
            || config
                .reply
                .as_ref()
                .is_some_and(|r| r.target_instance_id != self.profile.target_instance_id)
        {
            return Err(Px4ExternalModeArmingCheckError::TargetMismatch);
        }
        if config.profile.firmware_profile_id != self.profile.firmware_profile_id
            || config.request.firmware_profile_id != self.profile.firmware_profile_id
            || config
                .reply
                .as_ref()
                .is_some_and(|r| r.firmware_profile_id != self.profile.firmware_profile_id)
        {
            return Err(Px4ExternalModeArmingCheckError::FirmwareProfileMismatch);
        }
        if config.profile.source_boot_session_id != self.profile.source_boot_session_id
            || config.request.source_boot_session_id != self.profile.source_boot_session_id
            || config
                .reply
                .as_ref()
                .is_some_and(|r| r.source_boot_session_id != self.profile.source_boot_session_id)
        {
            return Err(Px4ExternalModeArmingCheckError::BootSessionMismatch);
        }
        if config.profile.receiver_clock_domain != self.profile.receiver_clock_domain
            || config.request.submitted_at.clock_domain != self.profile.receiver_clock_domain
            || config
                .reply
                .as_ref()
                .is_some_and(|r| r.received_at.clock_domain != self.profile.receiver_clock_domain)
        {
            return Err(Px4ExternalModeArmingCheckError::ReceiverClockMismatch);
        }
        if config.profile.producer_profile_id != config.request.producer_profile_id
            || config.profile.capture_window_id != config.request.capture_window_id
        {
            return Err(Px4ExternalModeArmingCheckError::SetpointConfigProfileMismatch);
        }
        let expected_profile = &self.profile.arming_check_producer_profile_id;
        if self.request.producer_profile_id != *expected_profile {
            return Err(Px4ExternalModeArmingCheckError::ArmingCheckProducerMismatch);
        }
        if self.registration.success {
            if self.registration.mode_id_raw < 0 {
                return Err(Px4ExternalModeArmingCheckError::SuccessfulRegistrationHasInvalidIds);
            }
            let mode_id = self.registration.mode_id_raw as u8;
            if config.request.source_id != mode_id {
                return Err(Px4ExternalModeArmingCheckError::ModeRegistrationMismatch);
            }
            if let Some(reply) = &config.reply {
                if reply.source_id != mode_id {
                    return Err(Px4ExternalModeArmingCheckError::ModeRegistrationMismatch);
                }
            }
        }
        Ok(())
    }

    fn check_request_profile(&self) -> Result<(), Px4ExternalModeArmingCheckError> {
        if self.request.target_instance_id != self.profile.target_instance_id {
            return Err(Px4ExternalModeArmingCheckError::TargetMismatch);
        }
        if self.request.firmware_profile_id != self.profile.firmware_profile_id {
            return Err(Px4ExternalModeArmingCheckError::FirmwareProfileMismatch);
        }
        if self.request.source_boot_session_id != self.profile.source_boot_session_id {
            return Err(Px4ExternalModeArmingCheckError::BootSessionMismatch);
        }
        if self.request.capture_window_id != self.profile.arming_check_capture_window_id {
            return Err(Px4ExternalModeArmingCheckError::CaptureWindowMismatch);
        }
        if self.request.received_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeArmingCheckError::ReceiverClockMismatch);
        }
        Ok(())
    }

    fn check_reply_profile(
        &self,
        reply: &Px4ExternalModeArmingCheckReplyV1,
    ) -> Result<(), Px4ExternalModeArmingCheckError> {
        if reply.producer_profile_id != self.profile.arming_check_producer_profile_id {
            return Err(Px4ExternalModeArmingCheckError::ArmingCheckProducerMismatch);
        }
        if reply.target_instance_id != self.profile.target_instance_id {
            return Err(Px4ExternalModeArmingCheckError::TargetMismatch);
        }
        if reply.firmware_profile_id != self.profile.firmware_profile_id {
            return Err(Px4ExternalModeArmingCheckError::FirmwareProfileMismatch);
        }
        if reply.source_boot_session_id != self.profile.source_boot_session_id {
            return Err(Px4ExternalModeArmingCheckError::BootSessionMismatch);
        }
        if reply.capture_window_id != self.profile.arming_check_capture_window_id {
            return Err(Px4ExternalModeArmingCheckError::CaptureWindowMismatch);
        }
        if reply.received_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ExternalModeArmingCheckError::ReceiverClockMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, &self.registration.registration_digest_hex);
        feed_str(&mut hasher, &self.setpoint_configuration.assessment_digest_hex);
        feed_str(&mut hasher, &self.request.request_digest_hex);
        match &self.reply {
            Some(reply) => {
                hasher.update(&[1]);
                feed_str(&mut hasher, &reply.reply_digest_hex);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Error validating or assessing the external-mode arming-check evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4ExternalModeArmingCheckError {
    /// Unsupported evidence schema version.
    #[error("unsupported arming-check evidence schema version {found}")]
    UnsupportedSchemaVersion { found: u16 },
    /// Unsupported PX4 message-definition version.
    #[error("unsupported PX4 message version {found}")]
    UnsupportedMessageVersion { found: u32 },
    /// Empty, padded, overlong, or control-containing identifier.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Registration reply supplied a raw ID below PX4's invalid sentinel.
    #[error("registration IDs below -1 are invalid")]
    InvalidRegistrationSentinel,
    /// A successful registration must assign component and mode IDs.
    #[error("successful registration has invalid mode or arming-check IDs")]
    SuccessfulRegistrationHasInvalidIds,
    /// Component registration ID cannot be represented by the PX4 32-bit mask.
    #[error("registration ID cannot be represented by the ArmingCheckRequest mask")]
    RegistrationIdOutsideMask,
    /// Profile freshness or response wait was zero.
    #[error("arming-check timing bounds must be non-zero")]
    ZeroTimingBound,
    /// Target identity differs across evidence.
    #[error("arming-check evidence target mismatch")]
    TargetMismatch,
    /// Firmware/API profile differs across evidence.
    #[error("arming-check evidence firmware-profile mismatch")]
    FirmwareProfileMismatch,
    /// PX4 boot/session identity differs across evidence.
    #[error("arming-check evidence boot-session mismatch")]
    BootSessionMismatch,
    /// Registration reply decoder differs from profile.
    #[error("registration evidence producer mismatch")]
    RegistrationProducerMismatch,
    /// Arming-check decoder differs from profile.
    #[error("arming-check evidence producer mismatch")]
    ArmingCheckProducerMismatch,
    /// Companion clock domain differs from profile.
    #[error("arming-check receiver-clock mismatch")]
    ReceiverClockMismatch,
    /// Assessment timestamp clock differs from profile.
    #[error("arming-check assessment-clock mismatch")]
    AssessmentClockMismatch,
    /// Arming-check request/reply capture windows differ from profile.
    #[error("arming-check capture-window mismatch")]
    CaptureWindowMismatch,
    /// SetpointConfig's mode ID did not match the mode ID in registration evidence.
    #[error("SetpointConfig mode ID does not match registered mode ID")]
    ModeRegistrationMismatch,
    /// Nested SetpointConfig profile did not match its source observations.
    #[error("SetpointConfig evidence profile does not match its source observations")]
    SetpointConfigProfileMismatch,
    /// The outcome classifier contradicted the computed SetpointConfig outcome.
    #[error("SetpointConfig outcome classification contradicted its source value")]
    OutcomeClassificationContradiction,
    /// ArmingCheckReply's event count exceeded the fixed Event array capacity.
    #[error("ArmingCheckReply num_events {found} exceeds the fixed event capacity of 5")]
    InvalidEventCount { found: u8 },
    /// Reply request_id did not match the captured ArmingCheckRequest.
    #[error("ArmingCheckReply request ID mismatch")]
    ReplyRequestIdMismatch,
    /// Reply registration_id did not match RegisterExtComponentReply.
    #[error("ArmingCheckReply registration ID mismatch")]
    ReplyRegistrationIdMismatch,
    /// PX4's source timestamp on the reply predates the correlated request timestamp.
    #[error("ArmingCheckReply PX4 source timestamp preceded its request")]
    ReplySourceTimestampRegression,
    /// Reply was received before its request.
    #[error("ArmingCheckReply receive time preceded its request")]
    ReceiveOrderRegression,
    /// Assessment predates evidence.
    #[error("arming-check assessment predates evidence")]
    AssessmentBeforeEvidence,
    /// A SetpointConfig reply is required after its profile reported acceptance.
    #[error("eligible SetpointConfig assessment omitted its reply")]
    MissingSetpointConfigReply,
    /// Nested SetpointConfig evidence was invalid.
    #[error("nested SetpointConfig assessment invalid: {0}")]
    SetpointConfig(#[from] Px4ExternalModeSetpointConfigError),
    /// Invalid timestamp or clock.
    #[error("invalid arming-check timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Stored content no longer matches its content commitment.
    #[error("arming-check {0} digest mismatch")]
    DigestMismatch(&'static str),
}

fn validate_schema(schema_version: u16) -> Result<(), Px4ExternalModeArmingCheckError> {
    if schema_version != PX4_EXTERNAL_MODE_ARMING_CHECK_SCHEMA_V1 {
        return Err(Px4ExternalModeArmingCheckError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4ExternalModeArmingCheckError> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4ExternalModeArmingCheckError::InvalidIdentifier(field));
    }
    Ok(())
}

fn map_assessment_timestamp_error(
    error: EvidenceValidationError,
) -> Px4ExternalModeArmingCheckError {
    match error {
        EvidenceValidationError::NonMonotonicTimestamp => {
            Px4ExternalModeArmingCheckError::AssessmentBeforeEvidence
        }
        other => Px4ExternalModeArmingCheckError::Timestamp(other),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external_mode_setpoint_config::{
        Px4ExternalModeSetpointConfigProfileV1, Px4ExternalModeSetpointConfigReplyV1,
        Px4ExternalModeSetpointConfigRequestV1,
    };

    fn clock() -> ClockDomainId {
        ClockDomainId::new("companion.monotonic").unwrap()
    }

    fn ts(ns: u64) -> TimestampV1 {
        TimestampV1::new(clock(), ns)
    }

    fn registration(
        success: bool,
        arming_check_id: i8,
        mode_id: i8,
    ) -> Px4ExternalModeRegistrationBindingV1 {
        Px4ExternalModeRegistrationBindingV1::new(
            "px4.registration.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            850_000,
            991,
            "org.luminous.fixture.mode",
            1,
            success,
            arming_check_id,
            mode_id,
            -1,
            false,
            ts(900_000_000),
        )
        .unwrap()
    }

    fn config_assessment(
        source_id: u8,
        should_apply: bool,
        result_raw: u8,
        assessed_ns: u64,
    ) -> Px4ExternalModeSetpointConfigAssessmentV1 {
        let window = "setpoint-config.fixture.attempt-1";
        let req = Px4ExternalModeSetpointConfigRequestV1::new(
            "px4.external-mode.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            window,
            1_000_000,
            4,
            source_id,
            should_apply,
            250,
            ts(800_000_000),
        )
        .unwrap();
        let reply = Px4ExternalModeSetpointConfigReplyV1::new(
            "px4.external-mode.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            window,
            1_050_000,
            4,
            source_id,
            result_raw,
            true,
            true,
            false,
            true,
            ts(850_000_000),
        )
        .unwrap();
        let profile = Px4ExternalModeSetpointConfigProfileV1::new(
            "px4.external-mode.setpoint-config.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.external-mode.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            window,
            clock(),
            200_000_000,
            500_000_000,
        )
        .unwrap();
        Px4ExternalModeSetpointConfigAssessmentV1::new(
            profile,
            req,
            Some(reply),
            ts(assessed_ns),
        )
        .unwrap()
    }

    fn arming_request(valid_mask: u32) -> Px4ExternalModeArmingCheckRequestV1 {
        Px4ExternalModeArmingCheckRequestV1::new(
            "px4.arming-check.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            "arming-check.fixture.cycle-1",
            1_200_000,
            8,
            valid_mask,
            ts(1_300_000_000),
        )
        .unwrap()
    }

    fn arming_reply(
        req_id: u8,
        registration_id: u8,
        local_alt: bool,
        received_ns: u64,
    ) -> Px4ExternalModeArmingCheckReplyV1 {
        arming_reply_with_diagnostics(
            req_id,
            registration_id,
            local_alt,
            received_ns,
            1_250_000,
            0,
            false,
            false,
            false,
            0,
            [
                Px4ExternalModeArmingCheckEventV1::default();
                PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn arming_reply_with_diagnostics(
        req_id: u8,
        registration_id: u8,
        local_alt: bool,
        received_ns: u64,
        source_timestamp_us_raw: u64,
        health_component_index: u8,
        health_component_is_present: bool,
        health_component_warning: bool,
        health_component_error: bool,
        num_events: u8,
        events: [Px4ExternalModeArmingCheckEventV1; PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1],
    ) -> Result<Px4ExternalModeArmingCheckReplyV1, Px4ExternalModeArmingCheckError> {
        Px4ExternalModeArmingCheckReplyV1::new(
            "px4.arming-check.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            "arming-check.fixture.cycle-1",
            source_timestamp_us_raw,
            req_id,
            registration_id,
            health_component_index,
            health_component_is_present,
            health_component_warning,
            health_component_error,
            true,
            num_events,
            events,
            true,
            true,
            local_alt,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            ts(received_ns),
        )
    }

    fn profile() -> Px4ExternalModeArmingCheckProfileV1 {
        Px4ExternalModeArmingCheckProfileV1::new(
            "px4.external-mode.arming-check.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.registration.decoder.fixture.v1",
            "px4.arming-check.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            "arming-check.fixture.cycle-1",
            clock(),
            2_000_000_000,
            300_000_000,
            200_000_000,
            500_000_000,
        )
        .unwrap()
    }

    fn assessment(
        reg: Px4ExternalModeRegistrationBindingV1,
        config: Px4ExternalModeSetpointConfigAssessmentV1,
        req: Px4ExternalModeArmingCheckRequestV1,
        reply: Option<Px4ExternalModeArmingCheckReplyV1>,
        assessed_ns: u64,
    ) -> Px4ExternalModeArmingCheckAssessmentV1 {
        Px4ExternalModeArmingCheckAssessmentV1::new(
            profile(),
            reg,
            config,
            req,
            reply,
            ts(assessed_ns),
        )
        .unwrap()
    }

    #[test]
    fn distinct_mode_and_registration_ids_bind_through_registration_evidence() {
        let value = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            1_400_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::RequirementsMatch {
                registration_id: 3,
                mode_id: 42,
                request_id: 8,
                can_arm_and_run: true,
            }
        ));
        assert!(value.assessment_id().unwrap().starts_with(
            "symthaea.autopilot.px4.arming-check-coherence-assessment.v1:"
        ));
    }

    #[test]
    fn independently_bound_registration_from_another_session_is_rejected() {
        let other_session = Px4ExternalModeRegistrationBindingV1::new(
            "px4.registration.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-b",
            850_000,
            991,
            "org.luminous.fixture.mode",
            1,
            true,
            3,
            42,
            -1,
            false,
            ts(900_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4ExternalModeArmingCheckAssessmentV1::new(
                profile(),
                other_session,
                config_assessment(42, true, 0, 1_200_000_000),
                arming_request(1 << 3),
                Some(arming_reply(8, 3, false, 1_350_000_000)),
                ts(1_400_000_000),
            )
            .unwrap_err(),
            Px4ExternalModeArmingCheckError::BootSessionMismatch
        );
    }

    #[test]
    fn mismatch_reports_only_differing_requirement_fields() {
        let value = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, true, 1_350_000_000)),
            1_400_000_000,
        );
        assert_eq!(
            value.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::RequirementsMismatch {
                differing_fields: vec![Px4ModeRequirementFieldV1::LocalAltitude],
                registration_id: 3,
                mode_id: 42,
                request_id: 8,
                can_arm_and_run: true,
            }
        );
    }

    #[test]
    fn can_arm_and_run_false_is_not_misreported_as_requirement_mismatch() {
        let reply = Px4ExternalModeArmingCheckReplyV1::new(
            "px4.arming-check.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.boot.fixture.session-a",
            "arming-check.fixture.cycle-1",
            1_250_000,
            8,
            3,
            0,
            false,
            false,
            false,
            false,
            0,
            [
                Px4ExternalModeArmingCheckEventV1::default();
                PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1
            ],
            true,
            true,
            false,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            ts(1_350_000_000),
        )
        .unwrap();
        let value = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(reply),
            1_400_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::RequirementsMatch {
                can_arm_and_run: false,
                ..
            }
        ));
    }

    #[test]
    fn registration_id_is_not_assumed_equal_to_mode_id() {
        let value = Px4ExternalModeArmingCheckAssessmentV1::new(
            profile(),
            registration(true, 3, 42),
            config_assessment(43, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            ts(1_400_000_000),
        );
        assert_eq!(value.unwrap_err(), Px4ExternalModeArmingCheckError::ModeRegistrationMismatch);
    }

    #[test]
    fn request_id_registration_id_and_valid_mask_are_independently_checked() {
        let wrong_request = Px4ExternalModeArmingCheckAssessmentV1::new(
            profile(),
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(9, 3, false, 1_350_000_000)),
            ts(1_400_000_000),
        );
        assert_eq!(
            wrong_request.unwrap_err(),
            Px4ExternalModeArmingCheckError::ReplyRequestIdMismatch
        );

        let wrong_registration = Px4ExternalModeArmingCheckAssessmentV1::new(
            profile(),
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 4, false, 1_350_000_000)),
            ts(1_400_000_000),
        );
        assert_eq!(
            wrong_registration.unwrap_err(),
            Px4ExternalModeArmingCheckError::ReplyRegistrationIdMismatch
        );

        let not_valid = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 4),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            1_400_000_000,
        );
        assert!(matches!(
            not_valid.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::RegistrationNotMarkedValid {
                registration_id: 3,
                ..
            }
        ));
    }

    #[test]
    fn reply_source_timestamp_before_request_fails_closed_despite_later_receive_time() {
        let reply = arming_reply_with_diagnostics(
            8,
            3,
            false,
            1_350_000_000,
            1_199_999,
            0,
            false,
            false,
            false,
            0,
            [
                Px4ExternalModeArmingCheckEventV1::default();
                PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1
            ],
        )
        .unwrap();
        let value = Px4ExternalModeArmingCheckAssessmentV1::new(
            profile(),
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(reply),
            ts(1_400_000_000),
        )
        .unwrap();
        assert_eq!(
            value.validate().unwrap_err(),
            Px4ExternalModeArmingCheckError::ReplySourceTimestampRegression
        );
    }

    #[test]
    fn absent_reply_is_not_promoted_to_component_failure() {
        let value = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            None,
            1_350_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::AwaitingReply { .. }
        ));
        let after_deadline = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            None,
            1_500_000_001,
        );
        assert!(matches!(
            after_deadline.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::ReplyNotObservedBeforeDeadline { .. }
        ));
    }

    #[test]
    fn failed_registration_and_stale_registration_fail_closed() {
        let failed = assessment(
            registration(false, -1, -1),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(0),
            None,
            1_400_000_000,
        );
        assert!(matches!(
            failed.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::RegistrationNotSuccessful
        ));

        let stale_profile = Px4ExternalModeArmingCheckProfileV1::new(
            "px4.external-mode.arming-check.stale-registration.fixture.v1",
            "vehicle.slot-a",
            "px4.v1.18.external-mode.fixture.v1",
            "px4.registration.decoder.fixture.v1",
            "px4.arming-check.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            "arming-check.fixture.cycle-1",
            clock(),
            2_000_000_000,
            5_000_000_000,
            200_000_000,
            500_000_000,
        )
        .unwrap();
        let stale = Px4ExternalModeArmingCheckAssessmentV1::new(
            stale_profile,
            Px4ExternalModeRegistrationBindingV1::new(
                "px4.registration.decoder.fixture.v1",
                "vehicle.slot-a",
                "px4.v1.18.external-mode.fixture.v1",
                "px4.boot.fixture.session-a",
                1,
                991,
                "org.luminous.fixture.mode",
                1,
                true,
                3,
                42,
                -1,
                false,
                ts(1),
            )
            .unwrap(),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            ts(4_000_000_000),
        )
        .unwrap();
        assert!(matches!(
            stale.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::RegistrationEvidenceStale { .. }
        ));
    }

    #[test]
    fn rejected_or_check_only_setpoint_config_cannot_qualify_arming_requirements() {
        let check_only = assessment(
            registration(true, 3, 42),
            config_assessment(42, false, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            1_400_000_000,
        );
        assert_eq!(
            check_only.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::SetpointConfigurationNotQualified {
                reason: Px4SetpointConfigBlockReasonV1::CheckOnlyRequest
            }
        );

        let rejected = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 2, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            1_400_000_000,
        );
        assert_eq!(
            rejected.outcome().unwrap(),
            Px4ExternalModeArmingCheckOutcomeV1::SetpointConfigurationNotQualified {
                reason: Px4SetpointConfigBlockReasonV1::ReplyRejected { result_raw: 2 }
            }
        );
    }

    #[test]
    fn mutating_capture_window_breaks_request_commitment() {
        let mut req = arming_request(1 << 3);
        req.capture_window_id = "arming-check.other-cycle".into();
        // The test exercises digest validation too: field mutation is detected before join.
        assert_eq!(
            assessment(
                registration(true, 3, 42),
                config_assessment(42, true, 0, 1_200_000_000),
                req,
                Some(arming_reply(8, 3, false, 1_350_000_000)),
                1_400_000_000,
            )
            .validate()
            .unwrap_err(),
            Px4ExternalModeArmingCheckError::DigestMismatch("arming-check request")
        );
    }

    #[test]
    fn serde_mutation_breaks_registration_and_assessment_commitments() {
        let value = assessment(
            registration(true, 3, 42),
            config_assessment(42, true, 0, 1_200_000_000),
            arming_request(1 << 3),
            Some(arming_reply(8, 3, false, 1_350_000_000)),
            1_400_000_000,
        );
        let encoded = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4ExternalModeArmingCheckAssessmentV1 =
            serde_json::from_slice(&encoded).unwrap();
        restored.validate().unwrap();
        restored.registration.mode_id_raw = 43;
        assert!(restored.validate().is_err());
    }

    #[test]
    fn reply_preserves_health_diagnostics_and_raw_event_payload_in_digest() {
        let mut events = [
            Px4ExternalModeArmingCheckEventV1::default();
            PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1
        ];
        events[0] = Px4ExternalModeArmingCheckEventV1 {
            timestamp_us_raw: 1_249_000,
            id: 0x1234,
            event_sequence: 7,
            arguments: core::array::from_fn(|index| index as u8),
            log_levels: 0xA5,
        };
        let reply = arming_reply_with_diagnostics(
            8, 3, false, 1_350_000_000, 1_250_000, 2, true, true, false, 1, events,
        )
        .unwrap();
        assert_eq!(reply.health_component_index, 2);
        assert!(reply.health_component_is_present);
        assert!(reply.health_component_warning);
        assert!(!reply.health_component_error);
        assert_eq!(reply.num_events, 1);
        assert_eq!(reply.events[0].timestamp_us_raw, 1_249_000);
        assert_eq!(reply.events[0].id, 0x1234);
        assert_eq!(reply.events[0].event_sequence, 7);
        assert_eq!(reply.events[0].arguments[24], 24);
        assert_eq!(reply.events[0].log_levels, 0xA5);
        reply.validate().unwrap();

        let mut mutated_health = reply.clone();
        mutated_health.health_component_warning = false;
        assert_eq!(
            mutated_health.validate().unwrap_err(),
            Px4ExternalModeArmingCheckError::DigestMismatch("arming-check reply")
        );

        let mut mutated_event = reply;
        mutated_event.events[0].arguments[24] ^= 1;
        assert_eq!(
            mutated_event.validate().unwrap_err(),
            Px4ExternalModeArmingCheckError::DigestMismatch("arming-check reply")
        );
    }

    #[test]
    fn event_count_above_px4_fixed_queue_capacity_fails_closed() {
        let error = arming_reply_with_diagnostics(
            8,
            3,
            false,
            1_350_000_000,
            1_250_000,
            0,
            false,
            false,
            false,
            (PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1 + 1) as u8,
            [
                Px4ExternalModeArmingCheckEventV1::default();
                PX4_ARMING_CHECK_REPLY_EVENT_CAPACITY_V1
            ],
        )
        .unwrap_err();
        assert_eq!(error, Px4ExternalModeArmingCheckError::InvalidEventCount { found: 6 });
    }

    #[test]
    fn zero_timing_bounds_fail_closed() {
        assert_eq!(
            Px4ExternalModeArmingCheckProfileV1::new(
                "px4.arming-check.invalid-profile.v1",
                "vehicle.slot-a",
                "px4.v1.18.external-mode.fixture.v1",
                "px4.registration.decoder.fixture.v1",
                "px4.arming-check.decoder.fixture.v1",
                "px4.boot.fixture.session-a",
                "arming-check.fixture.cycle-1",
                clock(),
                1_000,
                1_000,
                0,
                1_000,
            )
            .unwrap_err(),
            Px4ExternalModeArmingCheckError::ZeroTimingBound
        );
    }
}
